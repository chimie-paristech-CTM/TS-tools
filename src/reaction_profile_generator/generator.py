from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import numpy as np
import autode as ade
#from rdkit.Chem import rdMolAlign
import os
import subprocess
from autode.conformers import conf_gen
import shutil
from autode.conformers import conf_gen, Conformer
from typing import Callable
from functools import wraps

#xtb = ade.methods.XTB()

ps = Chem.SmilesParserParams()
ps.removeHs = False

workdir = 'test'

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

def get_distance(coord1, coord2):
    return np.sqrt((coord1 - coord2) ** 2)

@work_in(workdir)
def jointly_optimize_reactants_and_products(reactant_smiles, product_smiles):
    # get full smiles
    # determine active bonds
    # set bond lengths
    # Randomize-and-relax conformer generation
    # use same constraints to optimize with xTB
    # lift restrictions on the broken bonds -> reactants
    # lift restrictions on the formed bonds -> products
    #reactant_smiles = prepare_smiles(reactant_smiles)
    full_reactant_mol = Chem.MolFromSmiles(reactant_smiles, ps)
    full_product_mol = Chem.MolFromSmiles(product_smiles, ps)

    formed_bonds, broken_bonds = get_active_bonds(full_reactant_mol, full_product_mol)

    full_reactant_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in full_reactant_mol.GetAtoms()}

    formation_constraints, breaking_constraints = {}, {}
    print(reactant_smiles, product_smiles)
    print(formed_bonds, broken_bonds)
    for bond in formed_bonds:
       i,j = map(int, bond.split('-'))
       mol_begin = full_reactant_mol.GetAtomWithIdx(full_reactant_dict[i])
       mol_end = full_reactant_mol.GetAtomWithIdx(full_reactant_dict[j])
       formation_constraints[(mol_begin.GetIdx(), mol_end.GetIdx())] = \
        (covalent_radii_pm[mol_begin.GetAtomicNum()] + covalent_radii_pm[mol_end.GetAtomicNum()]) * 1.5 / 100

    for bond in broken_bonds:
        i,j = map(int, bond.split('-'))
        mol_begin = full_reactant_mol.GetAtomWithIdx(full_reactant_dict[i])
        mol_end = full_reactant_mol.GetAtomWithIdx(full_reactant_dict[j])
        breaking_constraints[(mol_begin.GetIdx(), mol_end.GetIdx())] = \
                                (covalent_radii_pm[mol_begin.GetAtomicNum()] + covalent_radii_pm[mol_end.GetAtomicNum()]) * 1.5 / 100

    get_conformer(full_reactant_mol)
    
    #coords = np.array(mol.GetConformer().GetPositions())

    write_xyz_file(full_reactant_mol, 'input.xyz')

    ade_mol = ade.Molecule('input.xyz', charge=-1)

    constraints = formation_constraints.copy()
    constraints.update(breaking_constraints)

    #constraints = dict(formation_constraints, **breaking_constraints)
    #constraints = merge_dicts(formation_constraints, breaking_constraints) #TODO: Fix!

    print(constraints)

    bonds = []
    for bond in full_reactant_mol.GetBonds():
        i,j = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        if (i,j) not in constraints and (j,i) not in constraints:
            #r0 = (covalent_radii_pm[full_reactant_mol.GetAtomWithIdx(i).GetAtomicNum()] + covalent_radii_pm[full_reactant_mol.GetAtomWithIdx(j).GetAtomicNum()]) /110
            bonds.append((i,j))
    
    ade_mol.graph.edges = bonds

    for n in range(5):
        atoms = conf_gen.get_simanl_atoms(species=ade_mol, dist_consts=constraints, conf_n=n)
        conformer = Conformer(name=f"conformer_{n}", atoms=atoms, charge=-1)

        conformer.optimise(method=ade.methods.XTB()) #TODO: this is not constrained optimization! Introduce now imposed activation!
        conformer.print_xyz_file()
    
    print(ade_mol.graph.edges) 

#def merge_dicts(dict1, dict2):
#    return(dict2.update(dict1))

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
