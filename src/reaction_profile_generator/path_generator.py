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
    def __init__(self, reactant_smiles, product_smiles, rxn_id, path_dir, rp_geometries_dir, solvent=None, reactive_complex_factor=2.0, freq_cut_off=150, n_conf=100):
        self.reactant_smiles = reactant_smiles
        self.product_smiles = product_smiles
        self.rxn_id = rxn_id
        self.path_dir = path_dir
        self.rp_geometries_dir = rp_geometries_dir
        self.solvent = solvent
        self.reactive_complex_factor = reactive_complex_factor
        self.freq_cut_off = freq_cut_off

        os.chdir(self.path_dir)

        self.reactant_rdkit_mol = Chem.MolFromSmiles(reactant_smiles, ps)
        self.product_rdkit_mol = Chem.MolFromSmiles(product_smiles, ps)
        self.charge = Chem.GetFormalCharge(self.reactant_rdkit_mol)
        self.formed_bonds, self.broken_bonds = self.get_active_bonds_from_mols()

        self.atom_map_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in self.reactant_rdkit_mol.GetAtoms()}
        self.atom_idx_dict = {atom.GetIdx(): atom.GetAtomMapNum() for atom in self.reactant_rdkit_mol.GetAtoms()}

        self.owning_dict_rsmiles = get_owning_mol_dict(reactant_smiles)
        self.owning_dict_psmiles = get_owning_mol_dict(product_smiles)

        self.formation_constraints = self.get_optimal_distances()

        self.stereo_correct_conformer_name = self.get_stereo_correct_conformer_name(n_conf)

        self.minimal_fc = self.determine_minimal_fc()

    def get_path(self):
        path_xyz_files = None
        if self.minimal_fc is not None:
            # overstretch range a bit because otherwise you may end up aborting the search prematurely
            for fc in np.arange(self.minimal_fc - 0.009, self.minimal_fc + 0.005, 0.001):
                reactive_complex_xyz_file = self.get_reactive_complex(min(fc, 0.1)) # you don't need very strong force constants to constrain non-covalently bounded reactants
                energies, coords, atoms, potentials = self.get_path_for_biased_optimization(reactive_complex_xyz_file, fc)
                if potentials[-1] > min(fc, 0.005):
                    continue  # Means that you haven't reached the products at the end of the biased optimization
                else:
                    if not self.endpoint_is_product(atoms, coords): 
                        print(f'Incorrect product formed for {self.rxn_id}')
                        return None, None, None
                    else:
                        path_xyz_files = get_path_xyz_files(atoms, coords, fc) 
                        self.save_rp_geometries(atoms, coords)
                        return energies, potentials, path_xyz_files
                
        return None, None, None

    def determine_minimal_fc(self):
        minimal_fc_crude = self.screen_fc_range(0.1, 4.0, 0.1)
        if minimal_fc_crude is not None:
            # overstretch range a bit because otherwise you may end up aborting the search prematurely
            minimal_fc_refined = self.screen_fc_range(minimal_fc_crude - 0.09, minimal_fc_crude + 0.03, 0.01)
        
        return minimal_fc_refined
    
    def screen_fc_range(self, start, end, interval):
        for fc in np.arange(start, end, interval):
            for _ in range(2):
                reactive_complex_xyz_file = self.get_reactive_complex(min(fc, 0.1))
                _, _, _, potentials = self.get_path_for_biased_optimization(reactive_complex_xyz_file, fc)
                if potentials[-1] < 0.005:
                    return fc
                else:
                    continue

        return None
    
    def get_formation_constraints_stretched(self):
        formation_constraints_to_stretch = self.get_bonds_to_stretch()
        formation_constraints_stretched = {x: random.uniform(self.reactive_complex_factor, 1.2 * self.reactive_complex_factor) * y 
                                       for x,y in self.formation_constraints.items() if x in formation_constraints_to_stretch}

        return formation_constraints_stretched
    
    def get_stereo_correct_conformer_name(self, n_conf=100):
        formation_constraints_stretched = self.get_formation_constraints_stretched()
        get_conformer(self.reactant_rdkit_mol)
    
        stereochemistry_smiles = find_stereocenters(self.reactant_rdkit_mol)
        write_xyz_file_from_mol(self.reactant_rdkit_mol, 'input_reactants.xyz', self.atom_map_dict)
        ade_mol = ade.Molecule(f'input_reactants.xyz', charge=self.charge)

        for node in ade_mol.graph.nodes:
            ade_mol.graph.nodes[node]['stereo'] = False

        bonds = []
        for bond in self.reactant_rdkit_mol.GetBonds():
            i,j = self.atom_map_dict[bond.GetBeginAtom().GetAtomMapNum()], self.atom_map_dict[bond.GetEndAtom().GetAtomMapNum()]

            if (i, j) not in formation_constraints_stretched and (j, i) not in formation_constraints_stretched:
                bonds.append((i,j))

        ade_mol.graph.edges = bonds

        # find good starting conformer
        for n in range(n_conf):
            atoms = conf_gen.get_simanl_atoms(species=ade_mol, dist_consts=formation_constraints_stretched, conf_n=n, save_xyz=False) # set save_xyz to false to ensure new optimization
            conformer = Conformer(name=f"conformer_reactants_init", atoms=atoms, charge=self.charge, dist_consts=formation_constraints_stretched)
            write_xyz_file_from_ade_atoms(atoms, f'{conformer.name}.xyz')
            stereochemistry_conformer = get_stereochemistry_from_conformer_xyz(f'{conformer.name}.xyz', self.reactant_smiles)

            if stereochemistry_smiles == stereochemistry_conformer:
                return conformer.name

        # print that there is an error with the stereochemistry only when you do a full search, i.e., n_conf > 1
        if n_conf > 1:
            print(f'No stereo-compatible conformer found for reaction {self.rxn_id}')

        return conformer.name

    def get_reactive_complex(self, fc):
        formation_constraints_stretched = self.get_formation_constraints_stretched()
        
        ade_mol_optimized = ade.Molecule(f'{self.stereo_correct_conformer_name}.xyz', charge=self.charge)

        ade_mol_optimized.constraints.update(formation_constraints_stretched)

        xtb.force_constant = fc
        ade_mol_optimized.optimise(method=xtb)

        write_xyz_file_from_ade_atoms(ade_mol_optimized.atoms, f'{self.stereo_correct_conformer_name}_opt.xyz')

        return os.path.join(f'{self.stereo_correct_conformer_name}_opt.xyz')

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

    def get_bonds_to_stretch(self):
        formation_bonds_to_stretch = set()

        for bond in self.formation_constraints.keys():
            if self.owning_dict_rsmiles[self.atom_idx_dict[bond[0]]] != self.owning_dict_rsmiles[self.atom_idx_dict[bond[1]]]:
                formation_bonds_to_stretch.add(bond)

        if len(formation_bonds_to_stretch) == 0:
            for bond in self.formation_constraints.keys():
                formation_bonds_to_stretch.add(bond) 

        return formation_bonds_to_stretch

    def get_active_bonds_from_mols(self):
        reactant_bonds = get_bonds(self.reactant_rdkit_mol)
        product_bonds = get_bonds(self.product_rdkit_mol)

        formed_bonds = product_bonds - reactant_bonds
        broken_bonds = reactant_bonds - product_bonds

        return formed_bonds, broken_bonds
    
    def get_optimal_distances(self):
        optimal_distances = {}
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
    
    def save_rp_geometries(self, atoms, coords):    
        write_xyz_file_from_atoms_and_coords(atoms[0], coords[0], os.path.join(self.rp_geometries_dir, 'reactants_geometry.xyz'))
        write_xyz_file_from_atoms_and_coords(atoms[-1], coords[-1], os.path.join(self.rp_geometries_dir, 'products_geometry.xyz'))

    def endpoint_is_product(self, atoms, coords):
        product_bonds = [(min(self.atom_map_dict[atom1], self.atom_map_dict[atom2]), max(self.atom_map_dict[atom1], self.atom_map_dict[atom2])) for atom1,atom2 in get_bonds(self.product_rdkit_mol)]
        write_xyz_file_from_atoms_and_coords(atoms[-1], coords[-1], 'products_geometry.xyz')
        ade_mol_p = ade.Molecule('products_geometry.xyz', name='products_geometry', charge=self.charge)

        if set(ade_mol_p.graph.edges) == set(product_bonds):
            return True
        else:
            return False
    

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


if __name__ == '__main__':
    shutil.rmtree('test')
    os.mkdir('test')
    os.chdir('test')
    #reactant_smiles = '[N+:1](=[B-:2](/[H:6])[H:7])(\[H:8])[H:9].[N+:3](=[B-:4](/[H:11])[H:12])(\[H:5])[H:10]'
    #product_smiles = '[N+:1]([B-:2]([H:6])([H:7])[H:12])([B:4]([N:3]([H:5])[H:10])[H:11])([H:8])[H:9]'
    #reactant_smiles = '[H:1]/[C:2](=[C:3](/[H:5])[O:6][H:7])[H:4].[O:8]=[C:9]([H:10])[H:11]'
    #product_smiles = '[H:1][C:2]([C:3]([H:5])=[O:6])([H:4])[C:9]([O:8][H:7])([H:10])[H:11]'
    #reactant_smiles = '[H:1]/[C:2](=[C:3](/[H:5])[O:6][H:7])[H:4].[O:8]=[C:9]([H:10])[H:11]'
    #product_smiles = '[H:1]/[C:2](=[C:3](\[O:6][H:7])[C:9]([O:8][H:5])([H:10])[H:11])[H:4]'
    reactant_smiles = '[H:1][O:2][C:3]([H:4])([H:5])[C@@:6]1([H:7])[O:8][C@@:9]([H:10])([O:11][H:12])[C@:13]([H:14])([O:15][H:16])[C@:17]1([H:18])[O:19][H:20].[H:21][O:22][P:23](=[O:24])([O-:25])[O:26][H:27]'
    product_smiles = '[O:25]([C@:9]1([H:10])[O:8][C@@:6]([C:3]([O:2][H:1])([H:4])[H:5])([H:7])[C@@:17]([H:18])([O:19][H:20])[C@@:13]1([H:14])[O:15][H:16])[P:23]([O:22][H:21])(=[O:24])[O:26][H:27].[O-:11][H:12]'
    reaction = PathGenerator(reactant_smiles, product_smiles, 'R1', 
                             '/Users/thijsstuyver/Desktop/reaction_profile_generator/path_test', 
                             '/Users/thijsstuyver/Desktop/reaction_profile_generator/rp_test')
    reaction.get_ts_guesses_from_path()
