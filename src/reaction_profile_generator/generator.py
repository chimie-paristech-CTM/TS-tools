from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import numpy as np
import autode as ade
from rdkit.Chem import rdMolAlign

xtb = ade.methods.XTB()

ps = Chem.SmilesParserParams()
ps.removeHs = False


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

    # get conformer for biggest fragment
    if product_mols[biggest_product_idx].GetNumAtoms() > reactant_mols[biggest_reactant_idx].GetNumAtoms():
       starting_mol = get_conformer(product_mols[biggest_product_idx])
       starting_mol_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in starting_mol.GetAtoms()}
       for i, mol in enumerate(reactant_mols):
            mol_to_align = get_conformer(mol)
            atom_maps = list(set([atom.GetAtomMapNum() for atom in mol_to_align.GetAtoms()]).intersection(
                set([atom.GetAtomMapNum() for atom in starting_mol.GetAtoms()])))
            mol_to_align_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol_to_align.GetAtoms()}
            id_mapping = [(mol_to_align_dict[i],starting_mol_dict[i]) for i in atom_maps]
            if len(atom_maps) != 0:
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

    AllChem.UFFOptimizeMolecule(full_reactant_mol)

    write_xyz_file(full_reactant_mol, 'reactant_final.xyz')

    #positions are scrambled!
    full_product_mol = get_conformer(full_product_mol)
    full_reactant_mol_dict_reverse = {atom.GetIdx(): atom.GetAtomMapNum() for atom in full_reactant_mol.GetAtoms()}
    full_product_mol_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in full_product_mol.GetAtoms()}

    product_conformer = full_product_mol.GetConformer()

    for i in range(full_product_mol.GetNumAtoms()):
        x,y,z = all_coord_r[i]
        product_conformer.SetAtomPosition(full_product_mol_dict[full_reactant_mol_dict_reverse[i]], Point3D(x,y,z)) 

    AllChem.UFFOptimizeMolecule(full_product_mol)

    write_xyz_file(full_product_mol, 'product_final.xyz')


def prepare_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
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

