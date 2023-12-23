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
import re
import shutil
import random

from reaction_profile_generator.utils import write_xyz_file_from_ade_atoms
from reaction_profile_generator.confirm_ts_guess import validate_ts_guess

ps = Chem.SmilesParserParams()
ps.removeHs = False
bohr_ang = 0.52917721090380

xtb = ade.methods.XTB()


class PathGenerator:
    def __init__(self, reactant_smiles, product_smiles, solvent=None, reactive_complex_factor=2.0, freq_cut_off=150):
        self.reactant_smiles = reactant_smiles
        self.product_smiles = product_smiles
        self.solvent = solvent
        self.reactive_complex_factor = reactive_complex_factor
        self.freq_cut_off = freq_cut_off

        self.reactant_rdkit_mol = Chem.MolFromSmiles(reactant_smiles, ps)
        self.product_rdkit_mol = Chem.MolFromSmiles(product_smiles, ps)
        self.charge = Chem.GetFormalCharge(self.reactant_rdkit_mol)
        self.formed_bonds, self.broken_bonds = self.get_active_bonds_from_mols()

        self.atom_map_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in self.reactant_rdkit_mol.GetAtoms()}
        self.atom_idx_dict = {atom.GetIdx(): atom.GetAtomMapNum() for atom in self.reactant_rdkit_mol.GetAtoms()}

        self.owning_dict_rsmiles = get_owning_mol_dict(reactant_smiles)
        self.owning_dict_psmiles = get_owning_mol_dict(product_smiles)

        self.formation_constraints = self.get_optimal_distances(reactant_side=False)
        self.breaking_constraints = self.get_optimal_distances(reactant_side=True)

        self.formation_bonds_to_stretch, self.breaking_bonds_to_stretch = self.get_bonds_to_stretch()

    def get_ts_guesses_from_path(self):
        minimal_fc = self.determine_minimal_fc()
        if minimal_fc is not None:
            for fc in np.arange(minimal_fc - 0.009, minimal_fc + 0.001, 0.001):
                for _ in range(5):
                    reactive_complex_xyz_file = self.get_reactive_complexes(fc)
                    energies, coords, atoms, potentials = self.get_path_for_biased_optimization(reactive_complex_xyz_file, fc)
                    if potentials[-1] > min(minimal_fc, 0.005):
                        break  # Means that you haven't reached the products at the end of the biased optimization
                    else:
                        save_rp_geometries(atoms, coords)
                        path_xyz_files = get_path_xyz_files(atoms, coords, fc)
                        guesses_found = get_ts_guesses(energies, potentials, path_xyz_files, \
                            self.charge, freq_cut_off=self.freq_cut_off)
                
                        if guesses_found:
                            return True
        
        return False

    def determine_minimal_fc(self):
        minimal_fc_crude = self.screen_fc_range(0.1, 0.8, 0.1)
        if minimal_fc_crude is not None:
            minimal_fc_refined = self.screen_fc_range(minimal_fc_crude - 0.09, minimal_fc_crude + 0.01, 0.01)
        
        return minimal_fc_refined
    
    # TODO: potentially check if bonding agrees with actual products as well + add constraints if problem
    def screen_fc_range(self, start, end, interval):
        for fc in np.arange(start, end, interval):
            reactive_complex_xyz_file = self.get_reactive_complexes(fc)
            _, _, _, potentials = self.get_path_for_biased_optimization(reactive_complex_xyz_file, fc)

            if potentials[-1] < 0.005:
                return fc
            else:
                continue

        return None

    def get_reactive_complexes(self, fc):
        formation_constraints_stretched = {x: random.uniform(self.reactive_complex_factor, 1.2 * self.reactive_complex_factor) * y 
                                       for x,y in self.formation_constraints.items() if x in self.formation_bonds_to_stretch}

        # Generate initial optimized reactant mol and conformer with the correct stereochemistry
        reactant_constraints = {} #breaking_constraints.copy()
        for key,val in formation_constraints_stretched.items():
            reactant_constraints[key] = val

        reactant_conformer_name = self.optimize_molecule_with_extra_constraints(reactant_constraints, fc)
    
        return reactant_conformer_name

    def get_path_for_biased_optimization(self, reactive_complex_xyz_file, fc):
        log_file = self.xtb_optimize_with_applied_potentials(reactive_complex_xyz_file, fc)
        all_energies, all_coords, all_atoms = read_energy_coords_file(log_file)

        valid_energies, valid_coords, valid_atoms = [], [], []
        for i, coords in enumerate(all_coords):
            valid_coords.append(coords)
            valid_atoms.append(all_atoms[i])
            valid_energies.append(all_energies[i])

        potentials = determine_potential(valid_coords, self.formation_constraints, fc)

        return valid_energies, valid_coords, valid_atoms, potentials

    # TODO: should this be with the formation constraints or only the bonds that actually undergo formation?
    def xtb_optimize_with_applied_potentials(self, reactive_complex_xyz_file, fc):
        xtb_input_path = f'{os.path.splitext(reactive_complex_xyz_file)[0]}.inp'

        with open(xtb_input_path, 'w') as f:
            f.write('$constrain\n')
            f.write(f'    force constant={fc}\n')
            for key, val in self.formation_constraints.items():
                f.write(f'    distance: {key[0] + 1}, {key[1] + 1}, {val}\n')
            f.write('$end\n')

        if self.solvent is not None:
            cmd = f'xtb {reactive_complex_xyz_file} --opt --input {xtb_input_path} -v --charge {self.charge} --solvent {self.solvent}'
        else:
            cmd = f'xtb {reactive_complex_xyz_file} --opt --input {xtb_input_path} -v --charge {self.charge}'

        with open(os.path.splitext(reactive_complex_xyz_file)[0] + '_path.out', 'w') as out:
            process = subprocess.Popen(cmd.split(), stderr=subprocess.DEVNULL, stdout=out)
            process.wait()

        os.rename('xtbopt.log', f'{os.path.splitext(reactive_complex_xyz_file)[0]}_path.log')

        return f'{os.path.splitext(reactive_complex_xyz_file)[0]}_path.log'

    def optimize_molecule_with_extra_constraints(self, constraints, fc, reactant_side=True):
        get_conformer(self.reactant_rdkit_mol)
    
        if reactant_side:
            stereochemistry_smiles = find_stereocenters(self.reactant_rdkit_mol)
            write_xyz_file_from_mol(self.reactant_rdkit_mol, 'input_reactants.xyz')
            ade_mol = ade.Molecule(f'input_reactants.xyz', charge=self.charge)
        else:
            stereochemistry_smiles = find_stereocenters(self.product_rdkit_mol)
            write_xyz_file_from_mol(self.reactant_rdkit_mol, 'input_products.xyz', self.atom_map_dict)
            ade_mol = ade.Molecule(f'input_products.xyz', charge=self.charge)

        for node in ade_mol.graph.nodes:
            ade_mol.graph.nodes[node]['stereo'] = False

        bonds = []
        for bond in self.reactant_rdkit_mol.GetBonds():
            if reactant_side:
                i, j = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
            else:
                i,j = self.atom_map_dict[bond.GetBeginAtom().GetAtomMapNum()], self.atom_map_dict[bond.GetEndAtom().GetAtomMapNum()]

            if (i, j) not in constraints and (j, i) not in constraints:
                bonds.append((i,j)) # TODO: double check that extra constraints are not needed here!

        ade_mol.graph.edges = bonds

        for n in range(20):
            atoms = conf_gen.get_simanl_atoms(species=ade_mol, dist_consts=constraints, conf_n=n, save_xyz=False) # set save_xyz to false to ensure new optimization
            
            if reactant_side:
                conformer = Conformer(name=f"conformer_reactant_init", atoms=atoms, charge=self.charge, dist_consts=constraints)
                write_xyz_file_from_ade_atoms(atoms, f'{conformer.name}.xyz')
                stereochemistry_conformer = get_stereochemistry_from_conformer_xyz(f'{conformer.name}.xyz', self.reactant_smiles)
            else:
                conformer = Conformer(name=f"conformer_product_init", atoms=atoms, charge=self.charge, dist_consts=constraints)
                write_xyz_file_from_ade_atoms(atoms, f'{conformer.name}.xyz')
                stereochemistry_conformer = get_stereochemistry_from_conformer_xyz(f'{conformer.name}.xyz', self.product_smiles)

            if stereochemistry_smiles == stereochemistry_conformer:
                break

        ade_mol_optimized = ade.Molecule(f'{conformer.name}.xyz', charge=self.charge)

        xtb_constraints = get_xtb_constraints(ade_mol_optimized, constraints)
        ade_mol_optimized.constraints.update(xtb_constraints)

        xtb.force_constant = fc
        ade_mol_optimized.optimise(method=xtb)

        write_xyz_file_from_ade_atoms(ade_mol_optimized.atoms, f'{conformer.name}.xyz')

        return f'{conformer.name}.xyz'         

    def get_bonds_to_stretch(self):
        formation_bonds_to_stretch, breaking_bonds_to_stretch = set(), set()

        for bond in set(self.formation_constraints.keys()) - set(self.breaking_constraints.keys()):
            if self.owning_dict_rsmiles[self.atom_idx_dict[bond[0]]] != self.owning_dict_rsmiles[self.atom_idx_dict[bond[1]]]:
                formation_bonds_to_stretch.add(bond)

        for bond in set(self.breaking_constraints.keys()) - set(self.formation_constraints.keys()):
            if self.owning_dict_psmiles[self.atom_idx_dict[bond[0]]] != self.owning_dict_psmiles[self.atom_idx_dict[bond[1]]]:
                breaking_bonds_to_stretch.add(bond)
        
        return formation_bonds_to_stretch, breaking_bonds_to_stretch

    def get_active_bonds_from_mols(self):
        reactant_bonds = get_bonds(self.reactant_rdkit_mol)
        product_bonds = get_bonds(self.product_rdkit_mol)

        formed_bonds = product_bonds - reactant_bonds
        broken_bonds = reactant_bonds - product_bonds

        return formed_bonds, broken_bonds
    
    def get_optimal_distances(self, reactant_side=True):
        optimal_distances = {}
        if reactant_side == True: 
            mols = [Chem.MolFromSmiles(smi, ps) for smi in self.reactant_smiles.split('.')]
            bonds = self.broken_bonds
            owning_mol_dict = self.owning_dict_rsmiles
        else:
            mols = [Chem.MolFromSmiles(smi, ps) for smi in self.product_smiles.split('.')]
            bonds = self.formed_bonds
            owning_mol_dict = self.owning_dict_psmiles

        for bond in bonds:
            i = int(bond[0])
            j = int(bond[1])
            idx1, idx2 = self.atom_map_dict[i], self.atom_map_dict[j]
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

            if self.solvent is not None:
                ade_mol = ade.Molecule('tmp.xyz', name='tmp', charge=charge, solvent_name=self.solvent)
            else:
                ade_mol = ade.Molecule('tmp.xyz', name='tmp', charge=charge)

            ade_mol.conformers = [conf_gen.get_simanl_conformer(ade_mol)]

            ade_mol.conformers[0].optimise(method=xtb)
            dist_matrix = distance_matrix(ade_mol.coordinates, ade_mol.coordinates)
            current_bond_length = dist_matrix[mol_dict[i], mol_dict[j]]

            optimal_distances[idx1, idx2] = current_bond_length
    
        return optimal_distances
    

def get_owning_mol_dict(smiles):
    mols = [Chem.MolFromSmiles(smi, ps) for smi in smiles.split('.')]
    owning_mol_dict = {}
    for idx, mol in enumerate(mols):
        for atom in mol.GetAtoms():
            owning_mol_dict[atom.GetAtomMapNum()] = idx

    return owning_mol_dict


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

        if atom_1 < atom_2:
            bonds.add((atom_1, atom_2))
        else:
            bonds.add((atom_2, atom_1))

    return bonds


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


def angstrom_to_bohr(distance_angstrom):
    """
    Convert distance in angstrom to bohr.

    Args:
        distance_angstrom (float): Distance in angstrom.

    Returns:
        float: Distance in bohr.
    """
    return distance_angstrom * 1.88973


def get_path_xyz_files(atoms, coords, force_constant):
    """
    
    """
    path_xyz_files = []
    folder_name = f'path_xyzs_{force_constant:.4f}'
    if folder_name in os.listdir():
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)

    for i in range(len(atoms)):
        filename = write_xyz_file_from_atoms_and_coords(
            atoms[i],
            coords[i],
                f'{folder_name}/path_{force_constant}_{i}.xyz'
            )
        path_xyz_files.append(filename)  

    return path_xyz_files


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


def save_rp_geometries(atoms, coords):
    path_name = os.path.join('/'.join(os.getcwd().split('/')[:-1]), 'rp_geometries')
    reaction_name = os.getcwd().split('/')[-1]

    if reaction_name not in os.listdir(path_name):
        os.makedirs(os.path.join(path_name, reaction_name))
    
    write_xyz_file_from_atoms_and_coords(atoms[0], coords[0], os.path.join(os.path.join(path_name, reaction_name), 'reactants_geometry.xyz'))
    write_xyz_file_from_atoms_and_coords(atoms[-1], coords[-1], os.path.join(os.path.join(path_name, reaction_name), 'products_geometry.xyz'))


def get_ts_guesses(energies, potentials, path_xyz_files, charge, freq_cut_off=150):
    """
    ...
    """
    true_energies = list(np.array(energies) - np.array(potentials))
    print(true_energies)

    # Find local maxima in path
    indices_local_maxima = find_local_max_indices(true_energies)

    # Validate the local maxima and store their energy values
    ts_guess_dict = {}
    idx_local_maxima_correct_mode = []
    for index in indices_local_maxima:
        ts_guess_file, _ = validate_ts_guess(path_xyz_files[index], os.getcwd(), freq_cut_off, charge)
        if ts_guess_file is not None:
            idx_local_maxima_correct_mode.append(index)
            ts_guess_dict[ts_guess_file] = true_energies[index] 

    print(indices_local_maxima)

    # If none of the local maxima could potentially lie on the correct mode, abort
    if len(idx_local_maxima_correct_mode) == 0:
        return False

    # Sort guesses based on energy
    sorted_guess_dict = sorted(ts_guess_dict.items(), key=lambda x: x[1], reverse=True)
    ranked_guess_files = [item[0] for item in sorted_guess_dict]

    print(ranked_guess_files)

    for index, guess_file in enumerate(ranked_guess_files):
        copy_final_guess_xyz(guess_file, index)

    # TODO: filter based on RMSD!

    return True

def find_local_max_indices(numbers):
    local_max_indices = []
    for i in range(len(numbers) - 2, 0, -1):
        if numbers[i] > numbers[i - 1] and numbers[i] > numbers[i + 1]:
            local_max_indices.append(i)
    return local_max_indices


def find_stereocenters(mol):
    stereocenters = []

    # Find chiral centers
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
    for atom_idx, center_info in chiral_centers:
        stereocenter = {
            'type': 'chirality',
            'position': mol.GetAtomWithIdx(atom_idx).GetAtomMapNum(),
            'descriptor': center_info
        }
        stereocenters.append(stereocenter)

    # Find cis/trans bonds
    for bond in mol.GetBonds():
        if bond.GetStereo() > 0:
            stereocenter = {
                'type': 'cis/trans',
                'position': set([bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()]),
                'descriptor': bond.GetStereo()
            }
            stereocenters.append(stereocenter)

    return stereocenters


def get_stereochemistry_from_conformer_xyz(xyz_file, smiles):
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

    stereochemistry = find_stereocenters(mol)

    return stereochemistry


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


def copy_final_guess_xyz(ts_guess_file, index):
    """
    Copies the final transition state guess XYZ file to a designated folder and renames it.

    Args:
        ts_guess_file (str): Path to the transition state guess XYZ file.
        ...

    Returns:
        None
    """
    path_name = os.path.join('/'.join(os.getcwd().split('/')[:-1]), 'final_ts_guesses')
    reaction_name = os.getcwd().split('/')[-1]

    if reaction_name not in os.listdir(path_name):
        os.makedirs(os.path.join(path_name, reaction_name))

    shutil.copy(ts_guess_file, os.path.join(path_name, reaction_name))
    os.rename(
        os.path.join(os.path.join(path_name, reaction_name), ts_guess_file.split('/')[-1]), 
        os.path.join(os.path.join(path_name, reaction_name), f'ts_guess_{index}.xyz')
    )


if __name__ == '__main__':
    shutil.rmtree('test')
    os.mkdir('test')
    os.chdir('test')
    #reactant_smiles = '[N+:1](=[B-:2](/[H:6])[H:7])(\[H:8])[H:9].[N+:3](=[B-:4](/[H:11])[H:12])(\[H:5])[H:10]'
    #product_smiles = '[N+:1]([B-:2]([H:6])([H:7])[H:12])([B:4]([N:3]([H:5])[H:10])[H:11])([H:8])[H:9]'
    #reactant_smiles = '[H:1]/[C:2](=[C:3](/[H:5])[O:6][H:7])[H:4].[O:8]=[C:9]([H:10])[H:11]'
    #product_smiles = '[H:1][C:2]([C:3]([H:5])=[O:6])([H:4])[C:9]([O:8][H:7])([H:10])[H:11]'
    reactant_smiles = '[H:1]/[C:2](=[C:3](/[H:5])[O:6][H:7])[H:4].[O:8]=[C:9]([H:10])[H:11]'
    product_smiles = '[H:1]/[C:2](=[C:3](\[O:6][H:7])[C:9]([O:8][H:5])([H:10])[H:11])[H:4]'
    reaction = PathGenerator(reactant_smiles, product_smiles)
    reaction.get_ts_guesses_from_path()
