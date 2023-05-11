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
import random

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

    # construct dicts to translate between map numbers idxs and vice versa
    full_reactant_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in full_reactant_mol.GetAtoms()}
    full_reactant_dict_reverse = {atom.GetIdx(): atom.GetAtomMapNum() for atom in full_reactant_mol.GetAtoms()}

    # TODO: clean up!!!
    # get the constraints for the initial FF conformer search
    formation_constraints, breaking_constraints = get_constraints(formed_bonds, broken_bonds, full_reactant_mol, full_reactant_dict, covalent_radii_pm)
    formation_constraints = determine_optimum_distances(product_smiles, full_reactant_dict_reverse, formation_constraints, charge=charge)
    breaking_constraints = determine_optimum_distances(reactant_smiles, full_reactant_dict_reverse, breaking_constraints, charge=charge)
    formation_constraints.update((x, (2.0 + random.uniform(0, 0.1)) * y) for x,y in formation_constraints.items())

    # construct autodE molecule object and perform constrained conformer search
    get_conformer(full_reactant_mol)
    write_xyz_file(full_reactant_mol, 'input.xyz')
    #TODO: stereochemistry??? -> this should be postprocessed by adding constraints like in autode
    ade_mol = ade.Molecule('input.xyz', charge=charge) 

    # combine constraints
    constraints = formation_constraints.copy()
    constraints.update(breaking_constraints)

    print(constraints)

    # set bonds
    bonds = []
    for bond in full_reactant_mol.GetBonds():
        i,j = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        if (i,j) not in constraints and (j,i) not in constraints:
            bonds.append((i,j))
    
    ade_mol.graph.edges = bonds

    # generate constrained FF conformers
    # TODO: presumably, the atoms get scrambled here and you get bonds that no longer match the atom-mapping!
    conformer_xyz_files = []

    atom_sites_to_be_frozen  = []
    for key in constraints.keys():
        atom_sites_to_be_frozen + [key[0], key[1]]
    atom_sites_to_be_frozen = np.array(list(set(atom_sites_to_be_frozen)))
    print(atom_sites_to_be_frozen)

    for n in range(5):
        atoms = conf_gen.get_simanl_atoms(species=ade_mol, dist_consts=constraints, conf_n=n)
        conformer = Conformer(name=f"conformer_{n}", atoms=atoms, charge=charge, dist_consts=constraints) # solvent_name='water'
        write_xyz_file2(atoms, f'{conformer.name}.xyz')
        optimized_xyz =  xtb_optimize_with_frozen_atoms(f'{conformer.name}.xyz', atom_sites_to_be_frozen)
        conformer_xyz_files.append(optimized_xyz)

    # prune conformers 
    # TODO: this function is currently broken!
    clusters = count_unique_conformers(conformer_xyz_files, full_reactant_mol)
    conformers_to_do = [conformer_xyz_files[cluster[0]] for cluster in clusters]

    # Apply attractive potentials and run optimization
    formation_constraints_final = determine_optimum_distances(product_smiles, full_reactant_dict_reverse, formation_constraints, charge=charge)

    print(breaking_constraints, formation_constraints)
    print(formation_constraints_final)

    for conformer in conformers_to_do:
        for force_constant in np.arange(0.03, 0.06, 0.01):
            log_file = xtb_optimize_with_applied_potentials(conformer, formation_constraints_final, force_constant)
            energies, coords, atoms = read_energy_coords_file(log_file)
            potentials = determine_potential(coords, formation_constraints_final, force_constant)

            print(force_constant, potentials)
            print(energies - potentials)
            if potentials[-1] > 0.01:
                print(distance_matrix(coords[-1],coords[-1]))
                print(atoms[-1])
                continue # means that you don't reach the products
            else:
                true_energy = list(energies - potentials)
                ts_guess_index = true_energy.index(max(true_energy))
                print(ts_guess_index)
                break

        for index in range(ts_guess_index, ts_guess_index + 6):
            print(force_constant, distance_matrix(coords[index], coords[index]))
            filename = write_xyz_file3(atoms[index], coords[index], f'ts_guess_{force_constant}.xyz')
            with open('lol.out', 'w') as out:
                process = subprocess.Popen(f'xtb {filename} --charge {charge} --hess'.split(), stdout=out)
                process.wait()
            n_freq = count_negative_frequencies('g98.out')
            if n_freq == 1:
                break


def count_negative_frequencies(filename):
    with open(filename, 'r') as file:
        for line in file:
            if line.strip().startswith('Frequencies --'):
                frequencies = line.strip().split()[2:]
                print(frequencies)
                return len([freq for freq in frequencies if float(freq) < 0])


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


def write_xyz_file3(atoms, coords, filename='ts_guess.xyz'):
    with open(filename, 'w') as f:
        f.write(f'{len(atoms)}\n')
        f.write("test \n")
        for i, coord in enumerate(coords):
            x, y, z = coord
            f.write(f"{atoms[i]} {x:.6f} {y:.6f} {z:.6f}\n")
    return filename

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
            f.write(f'    distance: {key[0] + 1}, {key[1] + 1}, {val}\n')
        f.write('$end\n')
    
    with open(os.path.splitext(xyz_file_path)[0] + '_path.out', 'w') as out:
        process = subprocess.Popen(f'xtb {xyz_file_path} --opt --input {xtb_input_path} -v --charge {charge}'.split(), stdout=out)
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
# TODO: This can be made more efficient...
def determine_optimum_distances(smiles, dict_reverse, constraints, solvent=None, charge=0):
    mols = [Chem.MolFromSmiles(smi, ps) for smi in smiles.split('.')]
    owning_mol_dict = {}
    for idx, mol in enumerate(mols):
        for atom in mol.GetAtoms():
            owning_mol_dict[atom.GetAtomMapNum()] = idx

    optimum_distances = {}

    for active_bond in constraints.keys():
        idx1,idx2 = active_bond
        i,j = dict_reverse[idx1], dict_reverse[idx2]
        print(i,j, smiles)
        if owning_mol_dict[i] == owning_mol_dict[j]:
            mol = copy.deepcopy(mols[owning_mol_dict[i]])
        else:
            raise KeyError
    
        mol_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
        [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]

        # detour needed to avoid reordering of the atoms by autodE
        get_conformer(mol)
        write_xyz_file(mol, 'tmp.xyz')

        #TODO: fix this
        if '-' in Chem.MolToSmiles(mol):
            charge = -1
        elif '+' in Chem.MolToSmiles(mol):
            charge = 1
        else:
            charge = 0

        if solvent != None:
            ade_rmol = ade.Molecule('tmp.xyz', name='tmp', charge=charge, solvent_name=solvent)
        else:
            ade_rmol = ade.Molecule('tmp.xyz', name='tmp', charge=charge)
        ade_rmol.populate_conformers(n_confs=1)

        ade_rmol.conformers[0].optimise(method=xtb)
        dist_matrix = distance_matrix(ade_rmol.conformers[0]._coordinates, ade_rmol.conformers[0]._coordinates)
        current_bond_length = dist_matrix[mol_dict[i], mol_dict[j]]

        optimum_distances[idx1,idx2] = current_bond_length
    
    return optimum_distances


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