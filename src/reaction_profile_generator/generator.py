from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import numpy as np
import autode as ade
#from rdkit.Chem import rdMolAlign
import os
from autode.conformers import conf_gen
from autode.conformers import conf_gen, Conformer
from typing import Callable
from functools import wraps
from autode.geom import calc_heavy_atom_rmsd
from autode.constraints import Constraints
from scipy.spatial import distance_matrix
import copy
import subprocess
from autode.values import ValueArray
from autode.hessians import Hessian
from abc import ABC
import math
from itertools import combinations

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
def jointly_optimize_reactants_and_products(reactant_smiles, product_smiles):
    # get the reactant and product mol
    full_reactant_mol = Chem.MolFromSmiles(reactant_smiles, ps)
    full_product_mol = Chem.MolFromSmiles(product_smiles, ps)

    # TODO: determine this from SMILES
    charge = 0  

    # TODO: right now only single bonds are considered -> maybe not a problem???
    formed_bonds, broken_bonds = get_active_bonds(full_reactant_mol, full_product_mol) 

    # construct dicts to translate between inidces and map numbers and vice versa
    full_reactant_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in full_reactant_mol.GetAtoms()}
    full_reactant_dict_reverse = {atom.GetIdx(): atom.GetAtomMapNum() for atom in full_reactant_mol.GetAtoms()}
    full_product_dict_reverse = {atom.GetIdx(): atom.GetAtomMapNum() for atom in full_product_mol.GetAtoms()}

    # get the constraints for the initial FF conformer search
    formation_constraints, breaking_constraints = get_constraints(formed_bonds, broken_bonds, full_reactant_mol, full_reactant_dict, covalent_radii_pm)

    # construct autodE molecule object and perform constrained conformer search
    get_conformer(full_reactant_mol)
    write_xyz_file(full_reactant_mol, 'input.xyz')
    ade_mol = ade.Molecule('input.xyz', charge=charge) #TODO: stereochemistry??? -> this should be postprocessed by adding constraints like in autode

    constraints = formation_constraints.copy()
    constraints.update(breaking_constraints)

    bonds = []
    for bond in full_reactant_mol.GetBonds():
        i,j = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        if (i,j) not in constraints and (j,i) not in constraints:
            bonds.append((i,j))
    
    ade_mol.graph.edges = bonds

    conformer_xyz_files = []
    for n in range(5):
        atoms = conf_gen.get_simanl_atoms(species=ade_mol, dist_consts=constraints, conf_n=n)
        conformer = Conformer(name=f"conformer_{n}", atoms=atoms, charge=charge, dist_consts=constraints) # solvent_name='water'
        write_xyz_file2(atoms, f'{conformer.name}.xyz')
        optimized_xyz =  xtb_optimize_with_frozen_atoms(f'{conformer.name}.xyz', np.array([0, 2, 3]))
        conformer_xyz_files.append(optimized_xyz)

    clusters = count_unique_conformers(conformer_xyz_files, full_reactant_mol)
    conformers_to_do = [conformer_xyz_files[cluster[0]] for cluster in clusters]

    for conformer in conformers_to_do:
        log_file = xtb_optimize_with_applied_potentials(conformer, formation_constraints, 0.15) # TODO: also add repulsive terms???
        energies, coords, atoms = read_energy_coords_file(log_file)

        potentials = determine_potential(coords, formation_constraints, 0.15)

        print(energies + potentials)
        print(potentials)
        print(energies)
        print(distance_matrix(coords[7], coords[7]))

        coords_true_ts = np.array([[0.42493, -0.25002, -0.00000],[-0.69481, 0.09111, 0.00000],[1.51701, -0.26492, 0.00000],[1.49188, 1.03614, -0.00000]])

        print(distance_matrix(coords_true_ts, coords_true_ts))


        # write new input -> constraining potential
        # execute xtb calculation -> get intermediate geometries and energies
        # determine potential energy -> subtract these from the full energy
        # determine saddle point   


def write_xyz_file2(atoms, filename):
    """
    Turn atoms into an xyz file with the given `filename`.
    """
    with open(filename, 'w') as f:
        # Write the number of atoms as the first line
        f.write(str(len(atoms)) + '\n')
        
        # Write a comment line
        f.write('Generated by write_xyz_file()\n')
        
        # Write the atom symbols and coordinates for each atom
        for atom in atoms:
            f.write(f'{atom.atomic_symbol} {atom.coord[0]:.6f} {atom.coord[1]:.6f} {atom.coord[2]:.6f}\n')
    

def xtb_optimize_with_frozen_atoms(xyz_file_path, frozen_atoms, charge=0, spin=1):
    """
    Perform an xTB optimization of the geometry in the given xyz file, keeping
    the specified atoms frozen in place, and return the path to the optimized
    geometry file.
    
    :param xyz_file_path: The path to the xyz file to optimize.
    :param frozen_atoms: A list of the indices of the atoms to freeze.
    :return: The path to the optimized xyz file.
    """
    # Create the xTB input file.
    xtb_input_path = os.path.splitext(xyz_file_path)[0] + '.inp'

    with open(xtb_input_path, 'w') as f:
        f.write(f'$fix \n    atoms: {",".join(list(map(str,frozen_atoms + 1)))}\n$end\n')
    
    with open(os.path.splitext(xyz_file_path)[0] + '.out', 'w') as out:
        process = subprocess.Popen(f'xtb {xyz_file_path} --opt --input {xtb_input_path} --charge {charge}'.split(), stdout=out)
        process.wait()

    os.rename('xtbopt.xyz', f'{os.path.splitext(xyz_file_path)[0]}_optimized.xyz')

    return f'{os.path.splitext(xyz_file_path)[0]}_optimized.xyz'


# TODO: set constraints to correct distances in product
def xtb_optimize_with_applied_potentials(xyz_file_path, constraints, force_constant, charge=0, spin=1):
    """
    """
    # Create the xTB input file.
    xtb_input_path = os.path.splitext(xyz_file_path)[0] + '.inp'

    with open(xtb_input_path, 'w') as f:
        f.write('$constrain\n')
        f.write(f'    force constant={force_constant}\n')
        for key, val in constraints.items():
            f.write(f'    distance: {key[0]}, {key[1]}, {val * 0.6}\n')
        f.write('$end\n')
    
    with open(os.path.splitext(xyz_file_path)[0] + '_path.out', 'w') as out:
        process = subprocess.Popen(f'xtb {xyz_file_path} --opt --input {xtb_input_path} --charge {charge}'.split(), stdout=out)
        process.wait()

    os.rename('xtbopt.log', f'{os.path.splitext(xyz_file_path)[0]}_path.log')

    return f'{os.path.splitext(xyz_file_path)[0]}_path.log'

# TODO: this function is broken!!!!
def count_unique_conformers(xyz_file_paths, full_reactant_mol):
    # Load the molecules from the xyz files
    molecules = []
    for xyz_file_path in xyz_file_paths:
        with open(xyz_file_path, 'r') as xyz_file:
            lines = xyz_file.readlines()
            num_atoms = int(lines[0])
            coords = [list(map(float,line.split()[1:])) for line in lines[2:num_atoms+2]]
            symbols = [line.split()[0] for line in lines[2:num_atoms+2]]
            mol = Chem.MolFromMolBlock(Chem.MolToMolBlock(full_reactant_mol))
            for i, atom in enumerate(mol.GetAtoms()):
                pos = coords[i]
                atom.SetProp('x', str(pos[0]))
                atom.SetProp('y', str(pos[1]))
                atom.SetProp('z', str(pos[2]))
            molecules.append(mol)

    # Calculate the RMSD between all pairs of molecules
    rmsd_matrix = np.zeros((len(molecules), len(molecules)))
    for i, j in combinations(range(len(molecules)), 2):
        rmsd = AllChem.GetBestRMS(molecules[i], molecules[j])
        rmsd_matrix[i, j] = rmsd
        rmsd_matrix[j, i] = rmsd
    
    print(rmsd_matrix)

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
            print(np.array(coords))
            all_coords.append(np.array(coords))
            all_atoms.append(atoms)
    return np.array(all_energies), all_coords, all_atoms


def determine_potential(all_coords, constraints, force_constant):
    potentials = []
    for coords in all_coords:
        potential = 0
        dist_matrix = distance_matrix(coords,coords)
        for key, val in constraints.items():
            actual_distance = dist_matrix[key[0],key[1]] - (val * 0.6)
            print(actual_distance, angstrom_to_bohr(actual_distance), force_constant * angstrom_to_bohr(actual_distance) ** 2)
            potential += force_constant * angstrom_to_bohr(actual_distance) ** 2
        potentials.append(potential)
    
    return np.array(potentials)



    #__________    
    conformer.optimise(method=xtb) # TODO: fix the atoms exactly and optimize everything else
    conformers.append(conformer)
    # THESE CONFORMERS ARE ACTUALLY ALREADY GOOD ENOUGH IN QUALITY!!!!! -> simply figure out wtf is wrong with xtb-gaussian, and/or just do freq calculation to decide constraint magnitude

    conformers = prune_conformers(conformers, rmsd_tol = 0.3) # angström
    for conformer in conformers:
        print(conformer.energy)

    xtb.force_constant = 2

    for i, conformer in enumerate(conformers):
        for k in np.arange(0.2, 0.5, 0.1):
            new_conf = conformer.copy()
            new_conf.name = f'conf_{i}_k_{round(k,1)}'
            new_conf.constraints = Constraints()
            for key, val in formation_constraints.items():
                new_conf.constraints.update({key: val * 0.6}) # TODO: put here the correct bond distances
            print(formation_constraints)
            new_conf.optimise(method = xtb)
                

def angstrom_to_bohr(distance_angstrom):
    """
    Convert distance in angstrom to bohr.
    """
    return distance_angstrom * 1.88973


def read_hessian(filename):
    with open(filename, 'r') as f:
        data = f.read()
    print(' '.join(data.split()))
    numbers = list(map(float, data.split()[1:]))
    return numbers

def make_square_array(numbers):
    n = np.sqrt(len(numbers))
    if int(n) != n:
        raise ValueError("Invalid number of elements for a square array.")
    A = [[0 for j in range(n)] for i in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1):
            A[i][j] = A[j][i] = numbers[k]
            k += 1
    return A


    #__________________________
    raise KeyError
    breaking_constraints_final = determine_forces(reactant_smiles, broken_bonds, full_reactant_dict, full_reactant_dict_reverse)
    formation_constraints_final = determine_forces(product_smiles, formed_bonds, full_reactant_dict, full_product_dict_reverse)

    for i, conformer in enumerate(conformers):
        new_conf = conformer.copy()
        new_conf.constraints = Constraints()
        new_conf.name = f'test_{i}'
        xtb.force_constant = 2
        for key, val in breaking_constraints_final.items(): # TODO: maybe this needs to happen earlier???
            new_conf.constraints.update({key: val[0] * 1.25})
        for key, val in formation_constraints_final.items():
            i == 0
            new_conf.constraints.update({key: val[0] * (1.40 - 0.2 * i)})
            i += 1
        new_conf.optimise(method = xtb)


    # THIS ENTIRE PART MAY NOT BE NEEDED!
        energies = []
        final_key, final_val = list(breaking_constraints_final.items())[0]
        for distance in range(10, 40):
            xtb.force_constant = final_val[1]
            final_conf = new_conf.copy()
            final_conf.constraints = Constraints()
            final_conf.name = f'final_{i}_{distance/100}'
            final_conf.constraints.update({final_key: final_val[0] + distance/100})
            xtb.force_constant = 2 * final_val[1] # TODO: fix this!!!
            final_conf.optimise(method = xtb)
            energies.append(final_conf.energy)
            
        print(energies, range(10, 30)[energies.index(max(energies))])
        #raise KeyError
                #dist_matrix = distance_matrix(conformer._coordinates, conformer._coordinates) # you probably want to ensure that bond distances are longer than product distances
                #print(dist_matrix)
                #print('')
        

def get_constraints(formed_bonds, broken_bonds, full_reactant_mol, full_reactant_dict, radii_dict):
    formation_constraints, breaking_constraints = {}, {}

    for bond in formed_bonds:
       i,j = map(int, bond.split('-'))
       mol_begin = full_reactant_mol.GetAtomWithIdx(full_reactant_dict[i])
       mol_end = full_reactant_mol.GetAtomWithIdx(full_reactant_dict[j])
       formation_constraints[(mol_begin.GetIdx(), mol_end.GetIdx())] = \
        (radii_dict[mol_begin.GetAtomicNum()] + radii_dict[mol_end.GetAtomicNum()]) * 1.7 / 100

    for bond in broken_bonds:
        i,j = map(int, bond.split('-'))
        atom_begin = full_reactant_mol.GetAtomWithIdx(full_reactant_dict[i])
        atom_end = full_reactant_mol.GetAtomWithIdx(full_reactant_dict[j])
        breaking_constraints[(atom_begin.GetIdx(), atom_end.GetIdx())] = \
                                (radii_dict[atom_begin.GetAtomicNum()] + radii_dict[atom_end.GetAtomicNum()]) * 1 / 100

    return formation_constraints, breaking_constraints


# you don't want the other molecules to be there as well
def determine_forces(smiles, active_bonds, full_reactant_dict, full_mol_dict_reverse):
    #print(smiles, constraints, full_mol_dict_reverse)
    mols = [Chem.MolFromSmiles(smi, ps) for smi in smiles.split('.')]
    owning_mol_dict = {}
    for idx, mol in enumerate(mols):
        for atom in mol.GetAtoms():
            owning_mol_dict[atom.GetAtomMapNum()] = idx

    constraints = {}

    xtb.force_constant = 5

    for bond in active_bonds:
        i,j = map(int, bond.split('-'))
        print(i,j, owning_mol_dict[i], owning_mol_dict[j], smiles)
        if owning_mol_dict[i] == owning_mol_dict[j]:
            mol = copy.deepcopy(mols[owning_mol_dict[i]])
        else:
            print("WTF")
            raise KeyError
    
        mol_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
        print(mol_dict)
        [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]

        # detour needed to avoid reordering of the atoms by autodE
        get_conformer(mol)
        write_xyz_file(mol, 'reactant.xyz')

        ade_rmol = ade.Molecule('reactant.xyz', name='lol', solvent_name='water')
        ade_rmol.populate_conformers(n_confs=1)

        ade_rmol.conformers[0].optimise(method=xtb)
        dist_matrix = distance_matrix(ade_rmol.conformers[0]._coordinates, ade_rmol.conformers[0]._coordinates)
        current_bond_length = dist_matrix[mol_dict[i], mol_dict[j]]
        x0 = np.linspace(current_bond_length - 0.05, current_bond_length + 0.05, 5)
        energies = stretch(ade_rmol.conformers[0], x0, (mol_dict[i], mol_dict[j])) # make sure this is ok
        
        p = np.polyfit(np.array(x0), np.array(energies), 2)

        force_constant = float(p[0] * 2 * bohr_ang) 
        constraints[(full_reactant_dict[i],full_reactant_dict[j])] = [current_bond_length, force_constant]
    
    return constraints


def stretch(conformer, x0, bond):
    energies = []

    for length in x0:
        conformer.constraints.update({bond: length})
        conformer.name = f'stretch_{length}'
        conformer.optimise(method=ade.methods.XTB())
        energies.append(conformer.energy)
    
    return energies


def prune_conformers(conformers, rmsd_tol):
    conformers = [conformer for conformer in conformers if conformer.energy != None]
    #idx_list = [idx for idx in enumerate(conformers)]
    #energies = [conformers[idx].energy for idx in idx_list]
    #for i, energy in enumerate(reversed(idxs))
    for idx in reversed(range(len(conformers) - 1)):
        conf = conformers[idx]
        if any(calc_heavy_atom_rmsd(conf.atoms, other.atoms) < rmsd_tol 
               for o_idx, other in enumerate(conformers) if o_idx != idx):
            del conformers[idx]
    
    return conformers


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
        #num_bonds = round(bond.GetBondTypeAsDouble())

        if atom_1 < atom_2:
            bonds.add(f'{atom_1}-{atom_2}') #-{num_bonds}')
        else:
            bonds.add(f'{atom_2}-{atom_1}') #-{num_bonds}')

    return bonds


def get_conformer(mol):
    #mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    # Perform UFF optimization
    AllChem.UFFOptimizeMolecule(mol)

    mol.GetConformer()

    return mol


covalent_radii_pm = [
    31.0,
    28.0,
    128.0,
    96.0,
    84.0,
    76.0,
    71.0,
    66.0,
    57.0,
    58.0,
    166.0,
    141.0,
    121.0,
    111.0,
    107.0,
    105.0,
    102.0,
    106.0,
    102.0,
    203.0,
    176.0,
    170.0,
    160.0,
    153.0,
    139.0,
    161.0,
    152.0,
    150.0,
    124.0,
    132.0,
    122.0,
    122.0,
    120.0,
    119.0,
    120.0,
    116.0,
    220.0,
    195.0,
    190.0,
    175.0,
    164.0,
    154.0,
    147.0,
    146.0,
    142.0,
    139.0,
    145.0,
    144.0,
    142.0,
    139.0,
    139.0,
    138.0,
    139.0,
    140.0,
    244.0,
    215.0,
    207.0,
    204.0,
    203.0,
    201.0,
    199.0,
    198.0,
    198.0,
    196.0,
    194.0,
    192.0,
    192.0,
    189.0,
    190.0,
    187.0,
    175.0,
    187.0,
    170.0,
    162.0,
    151.0,
    144.0,
    141.0,
    136.0,
    136.0,
    132.0,
    145.0,
    146.0,
    148.0,
    140.0,
    150.0,
    150.0,
]


def get_distance(coord1, coord2):
    return np.sqrt((coord1 - coord2) ** 2)


def write_xyz_file(mol, filename):
    # Generate conformer and retrieve coordinates
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