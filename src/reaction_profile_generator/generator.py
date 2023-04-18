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

    formed_bonds, broken_bonds = get_active_bonds(full_reactant_mol, full_product_mol) # right now only single bonds are considered

    full_reactant_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in full_reactant_mol.GetAtoms()}
    full_reactant_dict_reverse = {atom.GetIdx(): atom.GetAtomMapNum() for atom in full_reactant_mol.GetAtoms()}

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
        atom_begin = full_reactant_mol.GetAtomWithIdx(full_reactant_dict[i])
        atom_end = full_reactant_mol.GetAtomWithIdx(full_reactant_dict[j])
        breaking_constraints[(atom_begin.GetIdx(), atom_end.GetIdx())] = \
                                (covalent_radii_pm[atom_begin.GetAtomicNum()] + covalent_radii_pm[atom_end.GetAtomicNum()]) * 1.5 / 100

    get_conformer(full_reactant_mol)

    write_xyz_file(full_reactant_mol, 'input.xyz')

    ade_mol = ade.Molecule('input.xyz', charge=0) #TODO: stereochemistry???

    constraints = formation_constraints.copy()
    constraints.update(breaking_constraints)

    bonds = []
    for bond in full_reactant_mol.GetBonds():
        i,j = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        if (i,j) not in constraints and (j,i) not in constraints:
             bonds.append((i,j))
    
    ade_mol.graph.edges = bonds

    conformers = []
    for n in range(5):
        atoms = conf_gen.get_simanl_atoms(species=ade_mol, dist_consts=constraints, conf_n=n)
        conformer = Conformer(name=f"conformer_{n}", atoms=atoms, charge=0, solvent_name='water', dist_consts=constraints)

        conformer.optimise(method=xtb)
        conformers.append(conformer)

    conformers = prune_conformers(conformers, rmsd_tol = 0.1) # angström

    # get force constant for xtb
    #r_conf = conformers[0].copy()
    #r_conf.name = f'r_conf'
    #r_conf.constraints = Constraints()
    #for key, val in formation_constraints.items():
    #    r_conf.constraints.update({key: val * 3})
    #r_conf.optimise(method=ade.methods.XTB())
    breaking_constraints_final = determine_forces(reactant_smiles, breaking_constraints, full_reactant_dict_reverse)

    for i, conformer in enumerate(conformers):
        energies = []
        for distance in range(5, 90):
            new_conf = conformer.copy()
            new_conf.constraints = Constraints()
            for key, val in breaking_constraints_final.items(): # maybe this should switch positions in nested loop???
                new_conf.name = f'final_{i}_{distance/100}'
                new_conf.constraints.update({key: val[0] + distance/100})
            xtb.force_constant = list(breaking_constraints_final.values())[0][1] # TODO: fix this!!!
            new_conf.optimise(method = xtb)
            energies.append(new_conf.energy)
            
        print(energies, range(5, 90)[energies.index(max(energies))])
        raise KeyError
                #dist_matrix = distance_matrix(conformer._coordinates, conformer._coordinates) # you probably want to ensure that bond distances are longer than product distances
                #print(dist_matrix)
                #print('')
        





# you don't want the other molecules to be there as well
def determine_forces(reactant_smiles, breaking_constraints, full_reactant_dict_reverse):
    reactant_mols = [Chem.MolFromSmiles(smi, ps) for smi in reactant_smiles.split('.')]
    owning_mol_dict = {}
    for idx, mol in enumerate(reactant_mols):
        for atom in mol.GetAtoms():
            owning_mol_dict[atom.GetAtomMapNum()] = idx

    breaking_constraints_final = {}

    xtb.force_constant = 5

    for i,j in breaking_constraints:
        if owning_mol_dict[full_reactant_dict_reverse[i]] == owning_mol_dict[full_reactant_dict_reverse[j]]:
            mol = reactant_mols[owning_mol_dict[full_reactant_dict_reverse[i]]]
        else:
            print("WTF")
            raise KeyError
    
        mol_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
        [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]

        # detour needed to avoid reordering of the atoms by autodE
        get_conformer(mol)
        write_xyz_file(mol, 'reactant.xyz')

        ade_rmol = ade.Molecule('reactant.xyz', name='lol', solvent_name='water')
        ade_rmol.populate_conformers(n_confs=1)

        ade_rmol.conformers[0].optimise(method=xtb)
        dist_matrix = distance_matrix(ade_rmol.conformers[0]._coordinates, ade_rmol.conformers[0]._coordinates)
        current_bond_length = dist_matrix[mol_dict[full_reactant_dict_reverse[i]], mol_dict[full_reactant_dict_reverse[j]]]
        x0 = np.linspace(current_bond_length - 0.05, current_bond_length + 0.05, 5)
        energies = stretch(ade_rmol.conformers[0], x0, (mol_dict[full_reactant_dict_reverse[i]], mol_dict[full_reactant_dict_reverse[j]])) # make sure this is ok
        
        p = np.polyfit(np.array(x0), np.array(energies), 2)

        force_constant = float(p[0] * 2 * bohr_ang)
        print(force_constant)  
        breaking_constraints_final[(i,j)] = [current_bond_length, force_constant]
    
    return breaking_constraints_final


def stretch(conformer, x0, bond):
    energies = []

    for length in x0:
        conformer.constraints.update({bond: length})
        conformer.name = f'x_{length}'
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


def get_distance(coord1, coord2):
    return np.sqrt((coord1 - coord2) ** 2)