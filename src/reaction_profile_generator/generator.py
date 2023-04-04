from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import numpy as np
import autode as ade
from rdkit.Chem import rdMolAlign
import os
import subprocess
from autode.conformers import conf_gen
import shutil

#xtb = ade.methods.XTB()

ps = Chem.SmilesParserParams()
ps.removeHs = False


def get_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2)

def jointly_optimize_reactants_and_products(reactant_smiles, product_smiles):
    # find the biggest fragment
    # find bonds broken/formed
    # generate conformer for biggest fragment
    # look at which fragments on other side overlap
    # align overlapping atoms with the biggest fragment
    # align overlapping atoms of remaining fragments to these
    reactant_smiles = prepare_smiles(reactant_smiles)
    full_reactant_mol = Chem.MolFromSmiles(reactant_smiles, ps)
    full_product_mol = Chem.MolFromSmiles(product_smiles, ps)

    reactant_mols, product_mols = [], []
    for smi in reactant_smiles.split('.'):
        reactant_mols.append(Chem.MolFromSmiles(smi, ps))
    for smi in product_smiles.split('.'):
        product_mols.append(Chem.MolFromSmiles(smi, ps))
    
    biggest_reactant_idx = np.argmax([mol.GetNumAtoms() for mol in reactant_mols])
    biggest_product_idx = np.argmax([mol.GetNumAtoms() for mol in product_mols])

    all_coord_r = []
    atoms_to_fix = []

    # get conformer for biggest fragment
    if product_mols[biggest_product_idx].GetNumAtoms() > reactant_mols[biggest_reactant_idx].GetNumAtoms():
       starting_mol = get_conformer(product_mols[biggest_product_idx])
       starting_mol_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in starting_mol.GetAtoms()}
       for i, mol in enumerate(reactant_mols):
            mol_to_align = get_conformer(mol)
            atom_map = list(set([atom.GetAtomMapNum() for atom in mol_to_align.GetAtoms()]).intersection(
                set([atom.GetAtomMapNum() for atom in starting_mol.GetAtoms()])))
            atoms_to_fix += atom_map
            mol_to_align_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol_to_align.GetAtoms()}
            id_mapping = [(mol_to_align_dict[i],starting_mol_dict[i]) for i in atom_map]
            if len(atom_map) != 0:
                rdMolAlign.AlignMol(mol_to_align, starting_mol, atomMap=id_mapping, maxIters=1000)
                conformer = mol_to_align.GetConformer()
                coords = conformer.GetPositions()
                all_coord_r += list(coords)
            else:
                raise KeyError #TODO: continue aligning in a second round until all fragments are connected

    full_reactant_mol = get_conformer(full_reactant_mol)
    reactant_conformer = full_reactant_mol.GetConformer()
    for i in range(full_reactant_mol.GetNumAtoms()):
        x,y,z = all_coord_r[i]
        reactant_conformer.SetAtomPosition(i, Point3D(x,y,z))

    #AllChem.UFFOptimizeMolecule(full_reactant_mol)

    #write_xyz_file(full_reactant_mol, 'reactant_final.xyz')

    full_product_mol = get_conformer(full_product_mol)
    full_reactant_mol_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in full_reactant_mol.GetAtoms()}
    full_reactant_mol_dict_reverse = {atom.GetIdx(): atom.GetAtomMapNum() for atom in full_reactant_mol.GetAtoms()}
    full_product_mol_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in full_product_mol.GetAtoms()}
    full_product_mol_dict_reverse = {atom.GetIdx(): atom.GetAtomMapNum() for atom in full_product_mol.GetAtoms()}

    print(Chem.MolToSmiles(full_reactant_mol), full_reactant_mol_dict)
    print(Chem.MolToSmiles(full_product_mol), full_product_mol_dict, full_product_mol_dict_reverse)

    product_conformer = full_product_mol.GetConformer()

    for i in range(full_product_mol.GetNumAtoms()):
        x,y,z = all_coord_r[i]
        product_conformer.SetAtomPosition(full_product_mol_dict[full_reactant_mol_dict_reverse[i]], Point3D(x,y,z)) 

    AllChem.UFFOptimizeMolecule(full_product_mol)

    #write_xyz_file(full_product_mol, 'product_final.xyz')

    #autodE
    perform_bonded_repulsion_ff_optimization(full_reactant_mol, full_reactant_mol_dict, atoms_to_fix, 'full_reactant_final') 
    perform_bonded_repulsion_ff_optimization(full_product_mol, full_product_mol_dict, atoms_to_fix, 'full_product_final')

    #reorder the product atoms
    with open('full_product_final_opt.xyz', 'r') as f:
        lines = f.readlines()

    coord_lines = lines[2:]

    with open('full_product_final_opt_reordered.xyz', 'w') as f:
        f.write(lines[0])
        f.write(lines[1])
        for i in range(len(coord_lines)):
            f.write(coord_lines[full_product_mol_dict[full_reactant_mol_dict_reverse[i]]])

    command = f"xtb full_reactant_final_opt.xyz --path full_product_final_opt_reordered.xyz --input {os.path.join(os.getcwd(), 'path.inp')}"
    subprocess.check_call(
        command.split(),
        stdout=open("xtblog.txt", "w"),
        stderr=open(os.devnull, "w"),
    )


def perform_bonded_repulsion_ff_optimization(mol, mol_dict, atoms_to_fix, name):
    conformer = mol.GetConformer()
    coords = np.array(conformer.GetPositions())
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) 
                       for bond in mol.GetBonds()]

    idx_to_fix = [mol_dict[i] for i in atoms_to_fix]

    # if already bond, then do not fix
    distance_constraints = []
    fixed_bonds = []
    for idx1 in idx_to_fix:
        for idx2 in idx_to_fix:
            if idx1 == idx2:
                continue
            else:
                fixed_bonds.append((idx1, idx2))
                distance_constraints.append(1.2 * get_distance(coords[idx1], coords[idx2]))

    bond_distance_matrix = np.zeros((len(coords), len(coords)))

    for (i,j) in bonds:
        r0 = (covalent_radii_pm[mol.GetAtomWithIdx(i).GetAtomicNum()] + covalent_radii_pm[mol.GetAtomWithIdx(j).GetAtomicNum()]) /110
        bond_distance_matrix[i,j] = bond_distance_matrix[j,i] = float(r0)

    for k, (i,j) in enumerate(fixed_bonds):
        if (i,j) in bonds or (j,i) in bonds:
            continue
        else:
            bond_distance_matrix[i,j] = bond_distance_matrix[j,i] = distance_constraints[k]

    opt_coords, _ = conf_gen._get_coords_energy(coords, bonds, 1, 0.01, bond_distance_matrix, 1e-5, fixed_bonds, fixed_idxs=None)

    # do second optimization, but let the molecules disperse now
    bond_distance_matrix2 = np.zeros((len(coords), len(coords)))

    for (i,j) in bonds:
        r0 = (covalent_radii_pm[mol.GetAtomWithIdx(i).GetAtomicNum()] + covalent_radii_pm[mol.GetAtomWithIdx(j).GetAtomicNum()]) /110
        bond_distance_matrix2[i,j] = bond_distance_matrix2[j,i] = float(r0)

    opt_coords2, _ = conf_gen._get_coords_energy(opt_coords, bonds, 1, 0.01, bond_distance_matrix2, 1e-5, fixed_bonds=[], fixed_idxs=None)

    conformer = mol.GetConformer()

    for i in range(mol.GetNumAtoms()):
        x,y,z = opt_coords2[i]
        conformer.SetAtomPosition(i, Point3D(x,y,z))

    write_xyz_file(mol, f'{name}.xyz')

    settings_path = os.path.join(os.getcwd(), 'xtb.inp')

    with open(settings_path, 'w') as f:
        f.write('$constrain \n')
        f.write(f'  atoms: {",".join(list(map(str, range(len(opt_coords2)))))}\n')
        #f.write(f'  atoms: {",".join(list(map(str, idx_to_fix)))}\n')
        f.write('$end')

    command = f"xtb {name}.xyz --opt --input {settings_path} --alpb water"

    subprocess.check_call(
        command.split(),
        stdout=open("xtblog.txt", "w"),
        stderr=open(os.devnull, "w"),
    )

    shutil.move('xtbopt.xyz', f'{name}_opt.xyz')


def prepare_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles, ps)
    if '[H' not in smiles:
        mol = Chem.AddHs(mol)
    if mol.GetAtoms()[0].GetAtomMapNum() != 1:
        [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in mol.GetAtoms()]

    return Chem.MolToSmiles(mol)


def get_active_bonds(reactant_smiles, product_smiles):
    reactant_mol = Chem.MolFromSmiles(reactant_smiles)
    reactant_mol = Chem.AddHs(reactant_mol)
    [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in reactant_mol.GetAtoms()]
    product_mol = Chem.MolFromSmiles(product_smiles, ps)

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
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    # Perform UFF optimization
    AllChem.UFFOptimizeMolecule(mol)

    mol.GetConformer()

    return mol


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