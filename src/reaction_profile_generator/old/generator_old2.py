from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import autode as ade
import os
from autode.conformers import conf_gen
from autode.conformers import conf_gen, Conformer
from typing import Callable
from functools import wraps
from scipy.spatial import distance_matrix
import copy
import subprocess
from itertools import combinations
import random
import re

ps = Chem.SmilesParserParams()
ps.removeHs = False
bohr_ang = 0.52917721090380
workdir = 'test'

xtb = ade.methods.XTB()


def work_in(dir_ext: str) -> Callable:
    """Execute a function in a different directory"""

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):

            here = os.getcwd()
            dir_path = os.path.join(here, dir_ext)

            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)

            os.chdir(dir_path)
            try:
                result = func(*args, **kwargs)
            finally:
                os.chdir(here)

                if len(os.listdir(dir_path)) == 0:
                    os.rmdir(dir_path)

            return result

        return wrapped_function

    return func_decorator


@work_in(workdir)
def find_ts_guess(reactant_smiles, product_smiles, solvent=None, n_conf=5):
    # get the reactant and product mol
    full_reactant_mol = Chem.MolFromSmiles(reactant_smiles, ps)
    full_product_mol = Chem.MolFromSmiles(product_smiles, ps)

    charge =  Chem.GetFormalCharge(full_reactant_mol)

    formed_bonds, broken_bonds = get_active_bonds(full_reactant_mol, full_product_mol) 

    # construct dicts to translate between map numbers idxs and vice versa
    full_reactant_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in full_reactant_mol.GetAtoms()}

    # get the constraints for the initial FF conformer search
    formation_constraints = get_optimal_distances(product_smiles, full_reactant_dict, formed_bonds, solvent=solvent, charge=charge)
    breaking_constraints = get_optimal_distances(reactant_smiles, full_reactant_dict, broken_bonds, solvent=solvent, charge=charge)
    formation_constraints_stretched = formation_constraints.copy()
    formation_constraints_stretched.update((x, 2.0 * y) for x,y in formation_constraints_stretched.items())

    # combine constraints if multiple reactants
    constraints = breaking_constraints.copy()
    if len(reactant_smiles.split('.')) != 1:
        constraints.update(formation_constraints_stretched)

    # generate initial optimized conformer with the correct stereochemistry
    optimized_ade_mol = optimize_reactant_with_product_constraints(full_reactant_mol, reactant_smiles, constraints, 
                                                                   breaking_constraints, formation_constraints_stretched, 
                                                                   charge)

    # generate additional conformers
    conformers_to_do = generate_additional_conformers(optimized_ade_mol, full_reactant_mol, constraints, charge, solvent, n_conf)

    # apply attractive potentials and run optimization
    for conformer in conformers_to_do:
        # find the minimal force constant yielding the product upon application of the formation constraints and
        # use that one to generate a first TS guess
        for force_constant in np.arange(0.02, 0.20, 0.02):
            energies, coords, atoms, potentials = get_profile_for_biased_optimization(conformer, formation_constraints, 
                                                                                      force_constant, charge=charge, solvent=solvent)
            if potentials[-1] > 0.0001:
                continue # means that you haven't reached the products
            else:
                preliminary_ts_guess_index = get_ts_guess_index(force_constant, energies, potentials)
                break

        # if the preliminary TS guess has multiple imaginary frequencies, iterate through the next 6 optimization points and 
        # select the first point that yields only 1 imaginary frequency (if all have multiple imaginary frequencies, 
        # then return the preliminary guess)
        xyz_file_final_ts_guess = get_final_ts_guess_geometry(preliminary_ts_guess_index, atoms, coords, force_constant, charge)

        print(xyz_file_final_ts_guess)


def get_final_ts_guess_geometry(preliminary_ts_guess_index, atoms, coords, force_constant, charge):
    """
    Retrieves the final transition state (TS) guess geometry based on the given parameters.

    Args:
        preliminary_ts_guess_index (int): Index of the preliminary TS guess.
        atoms (list): List of atom objects.
        coords (list): List of coordinate objects.
        force_constant (float): Force constant value.
        charge (int): Charge value.

    Returns:
        str: Filename of the final TS guess geometry XYZ file.
    """
    for index in range(preliminary_ts_guess_index, preliminary_ts_guess_index + 6):
        filename = write_xyz_file_from_atoms_and_coords(
            atoms[index],
            coords[index],
            f'ts_guess_{force_constant}.xyz'
        )
        neg_freq = get_negative_frequencies(filename, charge)

        if index == preliminary_ts_guess_index:
            neg_freq_init = neg_freq 

        if len(neg_freq) == 1:
            return filename 
    
    # If no exit yet, then return the geometry at the original index
    print(neg_freq_init)
    filename = write_xyz_file_from_atoms_and_coords(
        atoms[preliminary_ts_guess_index],
        coords[preliminary_ts_guess_index],
        f'ts_guess_{force_constant}.xyz'
    )

    return filename


def get_negative_frequencies(filename, charge):
    with open('hess.out', 'w') as out:
            process = subprocess.Popen(f'xtb {filename} --charge {charge} --hess'.split(), stdout=out)
            process.wait()
    neg_freq = read_negative_frequencies('g98.out')
    
    return neg_freq


def get_ts_guess_index(force_constant, energies, potentials):
    true_energy = list(energies - potentials)
    ts_guess_index = true_energy.index(max(true_energy))

    return ts_guess_index


def get_profile_for_biased_optimization(conformer, formation_constraints, force_constant, charge, solvent):
    log_file = xtb_optimize_with_applied_potentials(conformer, formation_constraints, force_constant, charge=charge, solvent=solvent)
    energies, coords, atoms = read_energy_coords_file(log_file)
    potentials = determine_potential(coords, formation_constraints, force_constant)
    write_xyz_file_from_atoms_and_coords(atoms[-1], coords[-1], 'product_geometry_obtained.xyz')

    return energies, coords, atoms, potentials


def generate_additional_conformers(optimized_ade_mol, full_reactant_mol, constraints, charge, solvent, n_conf):
    # generate additional conformers
    conformer_xyz_files = []

    for n in range(n_conf):
        atoms = conf_gen.get_simanl_atoms(species=optimized_ade_mol, dist_consts=constraints, conf_n=n_conf)
        conformer = Conformer(name=f"conformer_{n}", atoms=atoms, charge=charge, dist_consts=constraints)
        write_xyz_file_from_ade_atoms(atoms, f'{conformer.name}.xyz')
        optimized_xyz = xtb_optimize(f'{conformer.name}.xyz', charge=charge, solvent=solvent)
        conformer_xyz_files.append(optimized_xyz)

    # prune conformers 
    clusters = count_unique_conformers(conformer_xyz_files, full_reactant_mol)
    conformers_to_do = [conformer_xyz_files[cluster[0]] for cluster in clusters]

    return conformers_to_do


def optimize_reactant_with_product_constraints(full_reactant_mol, reactant_smiles, constraints, charge):
    # construct autodE molecule object and perform constrained conformer search
    get_conformer(full_reactant_mol)
    write_xyz_file_from_mol(full_reactant_mol, 'input.xyz')

    ade_mol = ade.Molecule('input.xyz', charge=charge)
    for node in ade_mol.graph.nodes:
        ade_mol.graph.nodes[node]['stereo'] = False    

    # set bonds
    bonds = []
    for bond in full_reactant_mol.GetBonds():
        i,j = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        if (i,j) not in constraints and (j,i) not in constraints:
            bonds.append((i,j))
    
    ade_mol.graph.edges = bonds

    # generate constrained FF conformers -> first find a reasonable geometry, then you fix the stereochemistry, and then you generate conformers again!
    stereochemistry_smiles_reactants = get_stereochemistry_from_smiles(full_reactant_mol)
    #stereochemistry_smiles_products = get_stereochemistry_from_smiles(full_product_mol)
    
    for n in range(100):
        atoms = conf_gen.get_simanl_atoms(species=ade_mol, dist_consts=constraints, conf_n=n)
        conformer = Conformer(name="conformer_init", atoms=atoms, charge=charge, dist_consts=constraints)
        write_xyz_file_from_ade_atoms(atoms, f'{conformer.name}.xyz')
        embedded_mol, stereochemistry_xyz_reactants = get_stereochemistry_from_xyz(f'{conformer.name}.xyz', reactant_smiles)
        #stereochemistry_xyz_products = get_stereochemistry_from_xyz(f'{conformer.name}.xyz', product_smiles)
        if stereochemistry_smiles_reactants == stereochemistry_xyz_reactants:
            break

    embedded_mol = assign_cis_trans_from_geometry(embedded_mol, smiles_with_stereo=reactant_smiles)
    write_xyz_file_from_mol(embedded_mol, "conformer_init.xyz")

    # fix the stereochemistry in autode -> should be done automatically! (probably not for pi bonds)
    ade_mol_optimized = ade.Molecule('conformer_init.xyz')

    return ade_mol_optimized


#TODO: maybe product mol too?
def get_stereochemistry_from_smiles(reactant_mol):
    """
    Check if the stereochemistry is present in the reactant molecule SMILES.
    """
    stereochemistry = Chem.FindMolChiralCenters(reactant_mol)

    return stereochemistry


def find_cis_trans_elements(mol):
    cis_trans_elements = []
    
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            stereo = bond.GetStereo()
            if stereo == Chem.rdchem.BondStereo.STEREOZ or stereo == Chem.rdchem.BondStereo.STEREOE:
                cis_trans_elements.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), stereo))

    return cis_trans_elements


def add_xyz_conformer(smiles, xyz_file):
    # Load the molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles, ps)
    
    # Read the atomic coordinates from the XYZ file
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
        num_atoms = int(lines[0])
        coords = []
        symbols = []
        for i in range(2, num_atoms+2):
            line = lines[i].split()
            symbol = line[0]
            x, y, z = map(float, line[1:])
            symbols.append(symbol)
            coords.append((x, y, z))

    # Add the conformer to the molecule
    conformer = Chem.Conformer(num_atoms)
    for i, coord in enumerate(coords):
        conformer.SetAtomPosition(i, coord)
    mol.AddConformer(conformer)
    
    return mol


def get_stereochemistry_from_xyz(xyz_file, smiles):
    # Load the XYZ file using RDKit.
    mol = Chem.MolFromSmiles(smiles, ps)
    Chem.RemoveStereochemistry(mol)
    no_stereo_smiles = Chem.MolToSmiles(mol)
    mol = add_xyz_conformer(no_stereo_smiles, xyz_file)

    mol.GetConformer()

    Chem.AssignStereochemistryFrom3D(mol)

    stereochemistry = Chem.FindMolChiralCenters(mol)

    return mol, stereochemistry


def extract_atom_map_numbers(string):
    matches = re.findall(r'/\[[A-Za-z]+:(\d+)]', string)
    matches += re.findall(r'\\\[[A-Za-z]+:(\d+)]', string)
    
    return list(map(int, matches))


def assign_cis_trans_from_geometry(mol, smiles_with_stereo):
    cis_trans_elements = []
    mol_with_stereo = Chem.MolFromSmiles(smiles_with_stereo, ps)
    cis_trans_elements = find_cis_trans_elements(mol_with_stereo)
    involved_atoms = extract_atom_map_numbers(smiles_with_stereo)

    print(involved_atoms)

    # Iterate through the bonds
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            atomj_idx = bond.GetBeginAtomIdx()
            atomk_idx = bond.GetEndAtomIdx()

            # Get the conformer coordinates for the atoms
            conf = mol.GetConformer()
            neighbors_atomj = mol.GetAtomWithIdx(atomj_idx).GetNeighbors()
            neighbors_atomk = mol.GetAtomWithIdx(atomk_idx).GetNeighbors()
            atomi_idx = [atom.GetIdx() for atom in neighbors_atomj if atom.GetAtomMapNum() in involved_atoms][0]
            atoml_idx = [atom.GetIdx() for atom in neighbors_atomk if atom.GetAtomMapNum() in involved_atoms][0]

            if (atomj_idx, atomk_idx, Chem.rdchem.BondStereo.STEREOZ) in cis_trans_elements:
                angle = 0
            elif (atomj_idx, atomk_idx, Chem.rdchem.BondStereo.STEREOE) in cis_trans_elements:
                angle = 180
            else:
                raise KeyError

            #print(Chem.rdMolTransforms.GetDihedralDeg(conf, atomi_idx, atomj_idx, atomk_idx, atoml_idx))
            # get neighbors -> check sequence of atom map numbers -> find the correct one to determine the dihedral angle
            Chem.rdMolTransforms.SetDihedralDeg(conf, atomi_idx, atomj_idx, atomk_idx, atoml_idx, angle)

    return mol


def read_negative_frequencies(filename):
    with open(filename, 'r') as file:
        for line in file:
            if line.strip().startswith('Frequencies --'):
                frequencies = line.strip().split()[2:]
                negative_frequencies = [freq for freq in frequencies if float(freq) < 0]
                return negative_frequencies


def write_xyz_file_from_ade_atoms(atoms, filename):
    """
    Turn ade atoms object into an xyz file with the given `filename`.
    """
    with open(filename, 'w') as f:
        # Write the number of atoms as the first line
        f.write(str(len(atoms)) + '\n')
        
        # Write a comment line
        f.write('Generated by write_xyz_file()\n')
        
        # Write the atom symbols and coordinates for each atom
        for atom in atoms:
            f.write(f'{atom.atomic_symbol} {atom.coord[0]:.6f} {atom.coord[1]:.6f} {atom.coord[2]:.6f}\n')


def write_xyz_file_from_atoms_and_coords(atoms, coords, filename='ts_guess.xyz'):
    with open(filename, 'w') as f:
        f.write(f'{len(atoms)}\n')
        f.write("test \n")
        for i, coord in enumerate(coords):
            x, y, z = coord
            f.write(f"{atoms[i]} {x:.6f} {y:.6f} {z:.6f}\n")
    return filename


def xtb_optimize(xyz_file_path, charge=0, solvent=None):
    """
    Perform an xTB optimization of the geometry in the given xyz file 
    and return the path to the optimized geometry file.
    
    :param xyz_file_path: The path to the xyz file to optimize.
    :return: The path to the optimized xyz file.
    """
    if solvent != None:
        cmd = f'xtb {xyz_file_path} --opt --charge {charge} --solvent {solvent}'
    else:
        cmd = f'xtb {xyz_file_path} --opt --charge {charge}'
    with open(os.path.splitext(xyz_file_path)[0] + '.out', 'w') as out:
        print(cmd)
        process = subprocess.Popen(cmd.split(), stdout=out)
        process.wait()

    os.rename('xtbopt.xyz', f'{os.path.splitext(xyz_file_path)[0]}_optimized.xyz')

    return f'{os.path.splitext(xyz_file_path)[0]}_optimized.xyz'


def xtb_optimize_with_applied_potentials(xyz_file_path, constraints, force_constant, charge=0, solvent=None):
    """
    """
    # Create the xTB input file.
    xtb_input_path = os.path.splitext(xyz_file_path)[0] + '.inp'

    with open(xtb_input_path, 'w') as f:
        f.write('$constrain\n')
        f.write(f'    force constant={force_constant}\n')
        for key, val in constraints.items():
            f.write(f'    distance: {key[0] + 1}, {key[1] + 1}, {val}\n')
        f.write('$end\n')
    
    if solvent != None:
        cmd = f'xtb {xyz_file_path} --opt --input {xtb_input_path} -v --charge {charge} --solvent {solvent}'
    else:
        cmd = f'xtb {xyz_file_path} --opt --input {xtb_input_path} -v --charge {charge}'

    with open(os.path.splitext(xyz_file_path)[0] + '_path.out', 'w') as out:
        print(cmd)
        process = subprocess.Popen(cmd.split(), stdout=out)
        process.wait()

    os.rename('xtbopt.log', f'{os.path.splitext(xyz_file_path)[0]}_path.log')

    return f'{os.path.splitext(xyz_file_path)[0]}_path.log'


def count_unique_conformers(xyz_file_paths, full_reactant_mol):
    # Load the molecules from the xyz files
    molecules = []
    for xyz_file_path in xyz_file_paths:
        with open(xyz_file_path, 'r') as xyz_file:
            lines = xyz_file.readlines()
            num_atoms = int(lines[0])
            coords = [list(map(float,line.split()[1:])) for line in lines[2:num_atoms+2]]
            mol = Chem.Mol(full_reactant_mol)
            conformer = mol.GetConformer()
            for i in range(num_atoms):
                conformer.SetAtomPosition(i, coords[i])
            molecules.append(mol)
            
    # Calculate the RMSD between all pairs of molecules
    rmsd_matrix = np.zeros((len(molecules), len(molecules)))
    for i, j in combinations(range(len(molecules)), 2):
        rmsd = AllChem.GetBestRMS(molecules[i], molecules[j])
        rmsd_matrix[i, j] = rmsd
        rmsd_matrix[j, i] = rmsd

    # Cluster the molecules based on their RMSD
    clusters = []
    for i in range(len(molecules)):
        cluster_found = False
        for cluster in clusters:
            if all(rmsd_matrix[i, j] < 0.5 for j in cluster):
                cluster.append(i)
                cluster_found = True
                break
        if not cluster_found:
            clusters.append([i])

    return clusters


def read_energy_coords_file(file_path):
    all_energies = []
    all_coords = []
    all_atoms = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            # read energy value from line starting with "energy:"
            if len(lines[i].split()) == 1 and lines[i+1].strip().startswith("energy:"):
                energy_line = lines[i+1].strip()
                energy_value = float(energy_line.split()[1])
                all_energies.append(energy_value)
                i += 2
            else:
                raise ValueError(f"Unexpected line while reading energy value: {energy_line}")
            # read coordinates and symbols for next geometry
            coords = []; atoms = []
            while i < len(lines) and len(lines[i].split()) != 1:
                atoms.append(lines[i].split()[0])
                coords.append(np.array(list(map(float,lines[i].split()[1:]))))
                i += 1

            all_coords.append(np.array(coords))
            all_atoms.append(atoms)
    return np.array(all_energies), all_coords, all_atoms


def determine_potential(all_coords, constraints, force_constant):
    potentials = []
    for coords in all_coords:
        potential = 0
        dist_matrix = distance_matrix(coords,coords)
        for key, val in constraints.items():
            actual_distance = dist_matrix[key[0],key[1]] - val
            potential += force_constant * angstrom_to_bohr(actual_distance) ** 2
        potentials.append(potential)
    
    return potentials
                

def angstrom_to_bohr(distance_angstrom):
    """
    Convert distance in angstrom to bohr.
    """
    return distance_angstrom * 1.88973


def get_optimal_distances(smiles, mapnum_dict, bonds, solvent=None, charge=0):
    mols = [Chem.MolFromSmiles(smi, ps) for smi in smiles.split('.')]
    owning_mol_dict = {}
    for idx, mol in enumerate(mols):
        for atom in mol.GetAtoms():
            owning_mol_dict[atom.GetAtomMapNum()] = idx

    optimal_distances = {}

    for bond in bonds:
        i,j,_ = map(int, bond.split('-'))
        idx1, idx2 = mapnum_dict[i], mapnum_dict[j]
        if owning_mol_dict[i] == owning_mol_dict[j]:
            mol = copy.deepcopy(mols[owning_mol_dict[i]])
        else:
            raise KeyError
    
        mol_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
        [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]

        # detour needed to avoid reordering of the atoms by autodE
        get_conformer(mol)
        write_xyz_file_from_mol(mol, 'tmp.xyz')

        charge = Chem.GetFormalCharge(mol)

        if solvent != None:
            ade_rmol = ade.Molecule('tmp.xyz', name='tmp', charge=charge, solvent_name=solvent)
        else:
            ade_rmol = ade.Molecule('tmp.xyz', name='tmp', charge=charge)
        ade_rmol.populate_conformers(n_confs=1)

        ade_rmol.conformers[0].optimise(method=xtb)
        dist_matrix = distance_matrix(ade_rmol.coordinates, ade_rmol.coordinates)
        current_bond_length = dist_matrix[mol_dict[i], mol_dict[j]]

        optimal_distances[idx1,idx2] = current_bond_length
    
    return optimal_distances


def prepare_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles, ps)
    if '[H' not in smiles:
        mol = Chem.AddHs(mol)
    if mol.GetAtoms()[0].GetAtomMapNum() != 1:
        [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in mol.GetAtoms()]

    return Chem.MolToSmiles(mol)


def get_active_bonds(reactant_mol, product_mol):
    reactant_bonds = get_bonds(reactant_mol)
    product_bonds = get_bonds(product_mol)

    formed_bonds = product_bonds - reactant_bonds
    broken_bonds = reactant_bonds - product_bonds

    return formed_bonds, broken_bonds


def get_bonds(mol):
    bonds = set()
    for bond in mol.GetBonds():
        atom_1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
        atom_2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
        num_bonds = round(bond.GetBondTypeAsDouble())

        if atom_1 < atom_2:
            bonds.add(f'{atom_1}-{atom_2}-{num_bonds}')
        else:
            bonds.add(f'{atom_2}-{atom_1}-{num_bonds}')

    return bonds


def get_conformer(mol):
    #mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    # Perform UFF optimization
    AllChem.UFFOptimizeMolecule(mol)

    mol.GetConformer()

    return mol


def get_distance(coord1, coord2):
    return np.sqrt((coord1 - coord2) ** 2)


def write_xyz_file_from_mol(mol, filename):
    # Generate conformer and retrieve coordinates from mol
    conformer = mol.GetConformer()
    coords = conformer.GetPositions()

    # Write coordinates to XYZ file
    with open(filename, "w") as f:
        f.write(str(mol.GetNumAtoms()) + "\n")
        f.write("test \n")
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            symbol = atom.GetSymbol()
            x, y, z = coords[i]
            f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")
            