from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import autode as ade
import os
from autode.conformers import conf_gen
from autode.conformers import conf_gen, Conformer
from scipy.spatial import distance_matrix
import copy
import subprocess
from itertools import combinations
import re
import random

from reaction_profile_generator.utils import write_xyz_file_from_ade_atoms

ps = Chem.SmilesParserParams()
ps.removeHs = False
bohr_ang = 0.52917721090380

xtb = ade.methods.XTB()
xtb.force_constant = 2


def get_path(reactant_smiles, product_smiles, solvent=None, n_conf=5, fc_min=0.05, fc_max=0.9, fc_delta=0.10):
    """
    Finds a reaction path by applying an AFIR potential based on the given reactant and product SMILES strings.

    Args:
        reactant_smiles (str): SMILES string of the reactant.
        product_smiles (str): SMILES string of the product.
        solvent (str, optional): Solvent to consider during calculations. Defaults to None.
        n_conf (int, optional): Number of additional conformers to generate. Defaults to 5.
        fc_min (float, optional): Minimum force constant value. Defaults to 0.05.
        fc_max (float, optional): Maximum force constant value. Defaults to 0.9.
        fc_delta (float, optional): Increment value for force constant. Defaults to 0.10.

    Returns:
        None
    """
    # Get the reactant and product mol
    full_reactant_mol = Chem.MolFromSmiles(reactant_smiles, ps)
    full_product_mol = Chem.MolFromSmiles(product_smiles, ps)

    charge = Chem.GetFormalCharge(full_reactant_mol)

    formed_bonds, broken_bonds = get_active_bonds_from_mols(full_reactant_mol, full_product_mol) 

    # if more bonds are broken than formed, then reverse the reaction
    if len(formed_bonds) < len(broken_bonds):
        full_reactant_mol = Chem.MolFromSmiles(product_smiles, ps)
        full_product_mol = Chem.MolFromSmiles(reactant_smiles, ps)
        formed_bonds, broken_bonds = get_active_bonds_from_mols(full_reactant_mol, full_product_mol)

    # Construct dict to translate between map numbers and idxs
    full_reactant_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in full_reactant_mol.GetAtoms()}

    # Get the constraints for the initial FF conformer search -- randomize the distances for the constraints somewhat to sample diverse configurations
    formation_constraints = get_optimal_distances(product_smiles, full_reactant_dict, formed_bonds, solvent=solvent, charge=charge)
    breaking_constraints = get_optimal_distances(reactant_smiles, full_reactant_dict, broken_bonds, solvent=solvent, charge=charge)
    formation_bonds_to_stretch = set(formation_constraints.keys()) - set(breaking_constraints.keys())
    formation_constraints_stretched = formation_constraints.copy()
    formation_constraints_stretched.update((x, random.uniform(1.8, 2.3) * y) for x,y in formation_constraints_stretched.items() if x in formation_bonds_to_stretch)

    # Combine constraints if multiple reactants
    constraints = breaking_constraints.copy()
    if len(reactant_smiles.split('.')) != 1:
        for key, val in formation_constraints_stretched.items():
            if key not in constraints:
                constraints[key] = val

    # Generate initial optimized conformer with the correct stereochemistry
    optimized_ade_reactant_mol = optimize_molecule_with_extra_constraints(
        full_reactant_mol,
        reactant_smiles,
        constraints,
        charge
    )

    # generate a geometry for the product and save xyz -- randomize the distances for the constraints somewhat to sample diverse configurations
    breaking_bonds_to_stretch = set(breaking_constraints.keys()) - set(formation_constraints.keys())
    breaking_constraints_stretched = breaking_constraints.copy()
    breaking_constraints_stretched.update((x, random.uniform(1.8, 2.3) * y) for x, y in breaking_constraints_stretched.items() if x in breaking_bonds_to_stretch)
    product_constraints = formation_constraints.copy()
    if len(product_smiles.split('.')) != 1:
        for key,val in breaking_constraints_stretched.items():
            if key not in product_constraints:
                product_constraints[key] = val

    _ = optimize_molecule_with_extra_constraints(
        full_reactant_mol,
        product_smiles,
        product_constraints,
        charge,
        name='product', 
        reordering_dict=full_reactant_dict,
        extra_constraints=breaking_constraints
    )

    # Generate additional conformers for the reactants
    conformers_to_do = generate_additional_conformers(
        optimized_ade_reactant_mol,
        full_reactant_mol,
        constraints,
        charge,
        solvent,
        n_conf
    )

    # Store bond lengths for the inactive bonds
    inactive_bonds = get_inactive_bonds(full_reactant_mol, full_product_mol)
    dist_mat = distance_matrix(optimized_ade_reactant_mol.coordinates, optimized_ade_reactant_mol.coordinates)
    inactive_bond_mask = get_inactive_bond_mask(inactive_bonds, len(optimized_ade_reactant_mol.coordinates), full_reactant_dict)
    masked_dist_mat = 1.1 * dist_mat * inactive_bond_mask

    constraints_tmp = formation_constraints.copy()
    # Apply attractive potentials and run optimization
    for conformer in conformers_to_do:
        # Find the minimal force constant yielding the product upon application of the formation constraints
        for force_constant in np.arange(fc_min, fc_max, fc_delta):
            energies, coords, atoms, potentials = get_profile_for_biased_optimization(
                conformer,
                constraints_tmp,
                force_constant,
                inactive_bond_mask,
                masked_dist_mat,
                charge=charge,
                solvent=solvent,
            )

            if potentials[-1] > 0.01:
                continue  # Means that you haven't reached the products at the end of the biased optimization
            else:
                path_xyz_files = get_path_xyz_files(atoms, coords, force_constant)
                print(energies, potentials)
                true_energies = list(np.array(energies) - np.array(potentials))
                print(true_energies)
                raise KeyError
                return path_xyz_files, formation_constraints, breaking_constraints


def get_inactive_bonds(reactant_mol, product_mol):
    """
    Get the inactive bonds.

    Args:
        reactant_mol (Chem.Mol): Reactant molecule.
        product_mol (Chem.Mol): Product molecule.

    Returns:
        set: the unchanged bonds.
    """
    reactant_bonds = get_bonds(reactant_mol)
    product_bonds = get_bonds(product_mol)

    inactive_bonds = reactant_bonds.intersection(product_bonds)

    return inactive_bonds


def get_optimal_distances(smiles, mapnum_dict, bonds, solvent=None, charge=0):
    """
    Calculate the optimal distances for a set of bonds in a molecule.

    Args:
        smiles (str): SMILES representation of the molecule.
        mapnum_dict (dict): Dictionary mapping atom map numbers to atom indices.
        bonds (list): List of bond strings.
        solvent (str, optional): Name of the solvent. Defaults to None.
        charge (int, optional): Charge of the molecule. Defaults to 0.

    Returns:
        dict: Dictionary mapping bond indices to their corresponding optimal distances.
    """
    mols = [Chem.MolFromSmiles(smi, ps) for smi in smiles.split('.')]
    owning_mol_dict = {}
    for idx, mol in enumerate(mols):
        for atom in mol.GetAtoms():
            owning_mol_dict[atom.GetAtomMapNum()] = idx

    optimal_distances = {}

    for bond in bonds:
        i, j, _ = map(int, bond.split('-'))
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

        if solvent is not None:
            ade_rmol = ade.Molecule('tmp.xyz', name='tmp', charge=charge, solvent_name=solvent)
        else:
            ade_rmol = ade.Molecule('tmp.xyz', name='tmp', charge=charge)
        ade_rmol.populate_conformers(n_confs=1)

        ade_rmol.conformers[0].optimise(method=xtb)
        dist_matrix = distance_matrix(ade_rmol.coordinates, ade_rmol.coordinates)
        current_bond_length = dist_matrix[mol_dict[i], mol_dict[j]]

        optimal_distances[idx1, idx2] = current_bond_length
    
    return optimal_distances


def get_conformer(mol):
    """
    Generate and optimize a conformer of a molecule.

    Args:
        mol (Chem.Mol): Molecule.

    Returns:
        Chem.Mol: Molecule with optimized conformer.
    """
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    mol.GetConformer()

    return mol


def write_xyz_file_from_mol(mol, filename, reordering_dict=None):
    """
    Write a molecule's coordinates to an XYZ file.

    Args:
        mol (Chem.Mol): Molecule.
        filename (str): Name of the output XYZ file.
        reordering_dict (dict): dictionary to re-order the atoms
    """
    conformer = mol.GetConformer()
    coords = conformer.GetPositions()

    atom_info = [[] for _ in range(mol.GetNumAtoms())]   

    # reordering of the atoms may be needed
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        symbol = atom.GetSymbol()
        x, y, z = coords[i]
        if reordering_dict is not None:
            atom_info[reordering_dict[atom.GetAtomMapNum()]] = symbol, x, y, z
        else:
            atom_info[i] = symbol, x, y, z

    with open(filename, "w") as f:
        f.write(str(mol.GetNumAtoms()) + "\n")
        f.write("test \n")
        for i in range(mol.GetNumAtoms()):
            symbol, x, y, z = atom_info[i]
            f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")


# TODO: clean this up!
def run_hessian_calc(filename, charge):
    with open('hess.out', 'w') as out:
        process = subprocess.Popen(f'xtb {filename} --charge {charge} --hess'.split(), 
                                   stderr=subprocess.DEVNULL, stdout=out)
        process.wait()


# TODO: fix this!
def optimize_molecule_with_extra_constraints(full_mol, smiles, constraints, charge, name='reactant', reordering_dict=None, extra_constraints=None):
    """
    Optimize molecule with extra constraints.

    Args:
        full_mol: The full RDKIT molecule.
        smiles: SMILES representation of the molecule.
        constraints: Constraints for optimization.
        charge: Charge value.
        name: name to be used in generated xyz-files
        extra_constraints: undefined product constraints where there shouldn't be a bond

    Returns:
        object: The optimized ADE molecule.
    """
    get_conformer(full_mol)
    if reordering_dict is not None:
        write_xyz_file_from_mol(full_mol, f'input_{name}.xyz', reordering_dict)
    else:
        write_xyz_file_from_mol(full_mol, f'input_{name}.xyz')

    ade_mol = ade.Molecule(f'input_{name}.xyz', charge=charge)

    for node in ade_mol.graph.nodes:
        ade_mol.graph.nodes[node]['stereo'] = False

    bonds = []
    for bond in full_mol.GetBonds():
        if reordering_dict is not None:
            i, j = reordering_dict[bond.GetBeginAtom().GetAtomMapNum()], reordering_dict[bond.GetEndAtom().GetAtomMapNum()]
        else:
            i, j = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        if (i, j) not in constraints and (j, i) not in constraints:
            if extra_constraints != None:
                if (i,j) not in extra_constraints and (j,i) not in extra_constraints:
                    bonds.append((i, j))
            else:
                bonds.append((i, j))

    ade_mol.graph.edges = bonds

    stereochemistry_smiles_reactants = get_stereochemistry_from_mol(full_mol)

    for n in range(100):
        atoms = conf_gen.get_simanl_atoms(species=ade_mol, dist_consts=constraints, conf_n=n)
        conformer = Conformer(name=f"conformer_{name}_init", atoms=atoms, charge=charge, dist_consts=constraints)
        write_xyz_file_from_ade_atoms(atoms, f'{conformer.name}.xyz')
        embedded_mol, stereochemistry_xyz_reactants = get_stereochemistry_from_xyz(f'{conformer.name}.xyz', smiles)
        if stereochemistry_smiles_reactants == stereochemistry_xyz_reactants:
            break

    if len(stereochemistry_smiles_reactants) != 0:
        embedded_mol = assign_cis_trans_from_geometry(embedded_mol, smiles_with_stereo=smiles)
        write_xyz_file_from_mol(embedded_mol, f"conformer_{name}_init.xyz")

    ade_mol_optimized = ade.Molecule(f'conformer_{name}_init.xyz', charge=charge)

    xtb_constraints = get_xtb_constraints(ade_mol_optimized, constraints)
    conformer.constraints.update(xtb_constraints)

    ade_mol_optimized.constraints = conformer.constraints
    ade_mol_optimized.optimise(method=ade.methods.XTB())
    write_xyz_file_from_ade_atoms(ade_mol_optimized.atoms, f'conformer_{name}_init.xyz')

    return ade_mol_optimized


def get_stereochemistry_from_mol(mol):
    """
    Obtain stereochemistry present in a mol object.

    Args:
        mol: The RDKit molecule.

    Returns:
        list: The stereochemistry information.
    """
    stereochemistry = Chem.FindMolChiralCenters(mol)

    return stereochemistry


def get_profile_for_biased_optimization(conformer, formation_constraints, force_constant, inactive_bond_mask, dist_mat_mask, charge, solvent):
    """
    Retrieves the profile for biased optimization based on the given parameters.

    Args:
        conformer: The conformer object.
        formation_constraints: Constraints for formation.
        force_constant: Force constant value.
        charge: Charge value.
        solvent: Solvent to consider.
        ...

    Returns:
        tuple: A tuple containing coordinates, atoms, and potentials.
    """
    log_file = xtb_optimize_with_applied_potentials(conformer, formation_constraints, force_constant, charge=charge, solvent=solvent)
    all_energies, all_coords, all_atoms = read_energy_coords_file(log_file)

    # check if any of the inactive bonds has lengthened so much that the bond may have gotten dissociated (resulting in no gradient...),
    # and discard if this is the case
    valid_energies, valid_coords, valid_atoms = [], [], []
    for i, coords in enumerate(all_coords):
        curr_dist_mat = distance_matrix(coords,coords)
        bond_length_matrix = curr_dist_mat * inactive_bond_mask - dist_mat_mask
        if np.any(bond_length_matrix > 0):
            continue
        else:
            valid_coords.append(coords)
            valid_atoms.append(all_atoms[i])
            valid_energies.append(all_energies[i])

    potentials = determine_potential(valid_coords, formation_constraints, force_constant)
    write_xyz_file_from_atoms_and_coords(valid_atoms[-1], valid_coords[-1], 'product_geometry_obtained.xyz')

    return valid_energies, valid_coords, valid_atoms, potentials


def xtb_optimize_with_applied_potentials(xyz_file_path, constraints, force_constant, charge=0, solvent=None):
    """
    Perform an xTB optimization with applied potentials of the geometry in the given XYZ file and return the path to the log file.

    Args:
        xyz_file_path (str): The path to the XYZ file to optimize.
        constraints (dict): A dictionary specifying the atom index pairs and their corresponding distances.
        force_constant (float): The force constant to apply to the constraints.
        charge (int): The charge of the molecule (default: 0).
        solvent (str): The solvent to consider during the optimization (default: None).

    Returns:
        str: The path to the xTB log file.
    """
    xtb_input_path = os.path.splitext(xyz_file_path)[0] + '.inp'
    
    #print(constraints[(1,8)])
    try:
        #del constraints[(1,8)]
        constraints[(1,8)] = constraints[(1,8)] * 1.1
        print(constraints)
    except:
        pass


    with open(xtb_input_path, 'w') as f:
        f.write('$constrain\n')
        f.write(f'    force constant={force_constant}\n')
        for key, val in constraints.items():
            f.write(f'    distance: {key[0] + 1}, {key[1] + 1}, {val}\n')
        f.write('$end\n')

    if solvent is not None:
        cmd = f'xtb {xyz_file_path} --opt --input {xtb_input_path} -v --charge {charge} --solvent {solvent}'
    else:
        cmd = f'xtb {xyz_file_path} --opt --input {xtb_input_path} -v --charge {charge}'

    with open(os.path.splitext(xyz_file_path)[0] + '_path.out', 'w') as out:
        #print(cmd)
        process = subprocess.Popen(cmd.split(), stderr=subprocess.DEVNULL, stdout=out)
        process.wait()

    os.rename('xtbopt.log', f'{os.path.splitext(xyz_file_path)[0]}_path.log')

    return f'{os.path.splitext(xyz_file_path)[0]}_path.log'


def read_energy_coords_file(file_path):
    """
    Read energy and coordinate information from a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        Tuple: A tuple containing the energy values, coordinates, and atom symbols.
    """
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
            coords = []
            atoms = []
            while i < len(lines) and len(lines[i].split()) != 1:
                atoms.append(lines[i].split()[0])
                coords.append(np.array(list(map(float,lines[i].split()[1:]))))
                i += 1

            all_coords.append(np.array(coords))
            all_atoms.append(atoms)
    return np.array(all_energies), all_coords, all_atoms


def determine_potential(all_coords, constraints, force_constant):
    """
    Determine the potential energy for a set of coordinates based on distance constraints and a force constant.

    Args:
        all_coords (list): A list of coordinate arrays.
        constraints (dict): A dictionary specifying the atom index pairs and their corresponding distances.
        force_constant (float): The force constant to apply to the constraints.

    Returns:
        list: A list of potential energy values.
    """
    potentials = []
    for coords in all_coords:
        potential = 0
        dist_matrix = distance_matrix(coords, coords)
        for key, val in constraints.items():
            actual_distance = dist_matrix[key[0], key[1]] - val
            potential += force_constant * angstrom_to_bohr(actual_distance) ** 2
        potentials.append(potential)

    return potentials


def write_xyz_file_from_atoms_and_coords(atoms, coords, filename):
    """
    Write an XYZ file from a list of atoms and coordinates.

    Args:
        atoms: The list of atom symbols.
        coords: The list of atomic coordinates.
        filename: The name of the XYZ file to write.

    Returns:
        str: The name of the written XYZ file.
    """
    with open(filename, 'w') as f:
        f.write(f'{len(atoms)}\n')
        f.write("test \n")
        for i, coord in enumerate(coords):
            x, y, z = coord
            f.write(f"{atoms[i]} {x:.6f} {y:.6f} {z:.6f}\n")
    return filename


def angstrom_to_bohr(distance_angstrom):
    """
    Convert distance in angstrom to bohr.

    Args:
        distance_angstrom (float): Distance in angstrom.

    Returns:
        float: Distance in bohr.
    """
    return distance_angstrom * 1.88973


def get_xtb_constraints(ade_mol_optimized, constraints):
    """ """
    xtb_constraints = dict()
    dist_matrix = distance_matrix(ade_mol_optimized.atoms.coordinates, ade_mol_optimized.atoms.coordinates)
    active_atoms = set()
    for x,y in constraints.keys():
        active_atoms.add(x)
        active_atoms.add(y)
    for atom1 in list(active_atoms):
        for atom2 in list(active_atoms):
            if atom1 < atom2:
                xtb_constraints[(atom1, atom2)] = dist_matrix[atom1, atom2]
    
    return xtb_constraints


def get_path_xyz_files(atoms, coords, force_constant):
    """
    
    """
    path_xyz_files = []
    folder_name = f'path_xyzs_{force_constant}'
    os.makedirs(folder_name)

    for i in range(len(atoms)):
        filename = write_xyz_file_from_atoms_and_coords(
            atoms[i],
            coords[i],
                f'{folder_name}/path_{force_constant}_{i}.xyz'
            )
        path_xyz_files.append(filename)  

    return path_xyz_files


def generate_additional_conformers(optimized_ade_mol, full_reactant_mol, constraints, charge, solvent, n_conf):
    """
    Generate additional conformers based on the optimized ADE molecule and constraints.

    Args:
        optimized_ade_mol: The optimized ADE molecule.
        full_reactant_mol: The full reactant molecule.
        constraints: Constraints for conformer generation.
        charge: Charge value.
        solvent: Solvent to consider.
        n_conf: Number of additional conformers to generate.

    Returns:
        list: A list of additional conformers.
    """
    conformer_xyz_files = []

    for n in range(n_conf):
        atoms = conf_gen.get_simanl_atoms(species=optimized_ade_mol, dist_consts=constraints, conf_n=n_conf)
        conformer = Conformer(name=f"conformer_{n}", atoms=atoms, charge=charge, dist_consts=constraints)
        write_xyz_file_from_ade_atoms(atoms, f'{conformer.name}.xyz')
        optimized_xyz = xtb_optimize(f'{conformer.name}.xyz', charge=charge, solvent=solvent)
        conformer_xyz_files.append(optimized_xyz)

    clusters = count_unique_conformers(conformer_xyz_files, full_reactant_mol)
    conformers_to_do = [conformer_xyz_files[cluster[0]] for cluster in clusters]

    return conformers_to_do


def count_unique_conformers(xyz_file_paths, full_reactant_mol):
    """
    Count the number of unique conformers among a list of XYZ file paths based on RMSD clustering.

    Args:
        xyz_file_paths (list): A list of paths to the XYZ files.
        full_reactant_mol (Chem.Mol): The full reactant molecule as an RDKit Mol object.

    Returns:
        list: A list of clusters, where each cluster contains the indices of conformers belonging to the same cluster.
    """
    molecules = []
    for xyz_file_path in xyz_file_paths:
        with open(xyz_file_path, 'r') as xyz_file:
            lines = xyz_file.readlines()
            num_atoms = int(lines[0])
            coords = [list(map(float, line.split()[1:])) for line in lines[2:num_atoms+2]]
            mol = Chem.Mol(full_reactant_mol)
            conformer = mol.GetConformer()
            for i in range(num_atoms):
                conformer.SetAtomPosition(i, coords[i])
            molecules.append(mol)

    rmsd_matrix = np.zeros((len(molecules), len(molecules)))
    for i, j in combinations(range(len(molecules)), 2):
        rmsd = AllChem.GetBestRMS(molecules[i], molecules[j])
        rmsd_matrix[i, j] = rmsd
        rmsd_matrix[j, i] = rmsd

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

def xtb_optimize(xyz_file_path, charge=0, solvent=None):
    """
    Perform an xTB optimization of the geometry in the given XYZ file and return the path to the optimized geometry file.

    Args:
        xyz_file_path: The path to the XYZ file to optimize.
        charge: The charge of the molecule (default: 0).
        solvent: The solvent to consider during the optimization (default: None).

    Returns:
        str: The path to the optimized XYZ file.
    """
    if solvent is not None:
        cmd = f'xtb {xyz_file_path} --opt --charge {charge} --solvent {solvent}'
    else:
        cmd = f'xtb {xyz_file_path} --opt --charge {charge}'
    with open(os.path.splitext(xyz_file_path)[0] + '.out', 'w') as out:
        process = subprocess.Popen(cmd.split(), stderr=subprocess.DEVNULL, stdout=out)
        process.wait()

    os.rename('xtbopt.xyz', f'{os.path.splitext(xyz_file_path)[0]}_optimized.xyz')

    return f'{os.path.splitext(xyz_file_path)[0]}_optimized.xyz'


def get_stereochemistry_from_xyz(xyz_file, smiles):
    """
    Get stereochemistry information from an XYZ file.

    Args:
        xyz_file: The XYZ file.
        smiles: The SMILES string.

    Returns:
        object: The molecule with stereochemistry.
        list: The stereochemistry information.
    """
    mol = Chem.MolFromSmiles(smiles, ps)
    Chem.RemoveStereochemistry(mol)
    no_stereo_smiles = Chem.MolToSmiles(mol)
    mol = add_xyz_conformer(no_stereo_smiles, xyz_file)

    mol.GetConformer()

    Chem.AssignStereochemistryFrom3D(mol)

    stereochemistry = Chem.FindMolChiralCenters(mol)

    return mol, stereochemistry


def add_xyz_conformer(smiles, xyz_file):
    """
    Add an XYZ conformer to the molecule.

    Args:
        smiles: The SMILES string.
        xyz_file: The XYZ file.

    Returns:
        object: The molecule with the added conformer.
    """
    mol = Chem.MolFromSmiles(smiles, ps)
    
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

    conformer = Chem.Conformer(num_atoms)
    for i, coord in enumerate(coords):
        conformer.SetAtomPosition(i, coord)
    mol.AddConformer(conformer)
    
    return mol


def assign_cis_trans_from_geometry(mol, smiles_with_stereo):
    """
    Assign cis-trans configuration to the molecule based on the geometry.

    Args:
        mol: The RDKit molecule.
        smiles_with_stereo: The SMILES string with stereochemistry information.

    Returns:
        object: The molecule with assigned cis-trans configuration.
    """
    cis_trans_elements = []
    mol_with_stereo = Chem.MolFromSmiles(smiles_with_stereo, ps)
    cis_trans_elements = find_cis_trans_elements(mol_with_stereo)
    involved_atoms = extract_atom_map_numbers(smiles_with_stereo) # aren't these just the atoms of the double bond (j,k)???

    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            atomj_idx = bond.GetBeginAtomIdx()
            atomk_idx = bond.GetEndAtomIdx()
            conf = mol.GetConformer()
            neighbors_atomj = mol.GetAtomWithIdx(atomj_idx).GetNeighbors()
            neighbors_atomk = mol.GetAtomWithIdx(atomk_idx).GetNeighbors()
            try:
                atomi_idx = [atom.GetIdx() for atom in neighbors_atomj if atom.GetAtomMapNum() in involved_atoms][0]
                atoml_idx = [atom.GetIdx() for atom in neighbors_atomk if atom.GetAtomMapNum() in involved_atoms][0]
            except IndexError:
                continue

            if (atomj_idx, atomk_idx, Chem.rdchem.BondStereo.STEREOZ) in cis_trans_elements:
                angle = 0
            elif (atomj_idx, atomk_idx, Chem.rdchem.BondStereo.STEREOE) in cis_trans_elements:
                angle = 180
            else:
                raise KeyError

            Chem.rdMolTransforms.SetDihedralDeg(conf, atomi_idx, atomj_idx, atomk_idx, atoml_idx, angle)

    return mol


def find_cis_trans_elements(mol):
    """
    Find cis-trans elements in the molecule.

    Args:
        mol: The molecule.

    Returns:
        list: The cis-trans elements.
    """
    cis_trans_elements = []
    
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            stereo = bond.GetStereo()
            if stereo == Chem.rdchem.BondStereo.STEREOZ or stereo == Chem.rdchem.BondStereo.STEREOE:
                cis_trans_elements.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), stereo))

    return cis_trans_elements

def extract_atom_map_numbers(string):
    """
    Extract atom map numbers from a string.

    Args:
        string: The input string.

    Returns:
        list: The extracted atom map numbers.
    """
    matches = re.findall(r'/\[[A-Za-z]+:(\d+)]', string)
    matches += re.findall(r'\\\[[A-Za-z]+:(\d+)]', string)
    
    return list(map(int, matches))


def get_inactive_bond_mask(inactive_bonds, coordinates, full_reactant_dict):
    """

    Args:
        inactive_bonds (_type_): _description_
        coordinates (_type_): _description_
    """
    one_hot_array = np.zeros((coordinates, coordinates), dtype=int)

    # Set the corresponding indices to 1
    for inactive_bond in inactive_bonds:
        i,j,_ = map(int,inactive_bond.split('-'))
        one_hot_array[full_reactant_dict[i]][full_reactant_dict[j]] = 1
        one_hot_array[full_reactant_dict[j]][full_reactant_dict[i]] = 1

    return one_hot_array

def get_active_bonds_from_smiles(reactant_smiles, product_smiles):
    """
    Get the active bonds (formed and broken) between two SMILES.

    Args:
        reactant_smiles (string): Reactant string.
        product_smiles (string): Product string.

    Returns:
        tuple: A tuple containing two sets:
            - Formed bonds (set of bond strings).
            - Broken bonds (set of bond strings).
    """
    full_reactant_mol = Chem.MolFromSmiles(reactant_smiles, ps)
    full_product_mol = Chem.MolFromSmiles(product_smiles, ps)

    formed_bonds, broken_bonds = get_active_bonds_from_mols(full_reactant_mol, full_product_mol)

    return formed_bonds, broken_bonds 


def get_active_bonds_from_mols(reactant_mol, product_mol):
    """
    Get the active bonds (formed and broken) between two molecules.

    Args:
        reactant_mol (Chem.Mol): Reactant molecule.
        product_mol (Chem.Mol): Product molecule.

    Returns:
        tuple: A tuple containing two sets:
            - Formed bonds (set of bond strings).
            - Broken bonds (set of bond strings).
    """
    reactant_bonds = get_bonds(reactant_mol)
    product_bonds = get_bonds(product_mol)

    formed_bonds = product_bonds - reactant_bonds
    broken_bonds = reactant_bonds - product_bonds

    return formed_bonds, broken_bonds


def get_bonds(mol):
    """
    Get the bond strings of a molecule.

    Args:
        mol (Chem.Mol): Molecule.

    Returns:
        set: Set of bond strings.
    """
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
