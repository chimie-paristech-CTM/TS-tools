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
from autode.values import ValueArray
from autode.hessians import Hessian
from autode.transition_states import TSbase

ps = Chem.SmilesParserParams()
ps.removeHs = False
bohr_ang = 0.52917721090380

xtb = ade.methods.XTB()


# get ade reactant from xyz
# get ade product from xyz
# define instance of TSbase class
# set hessian for the TSbase instance -> see below
# apply has_correct_imag_modes method from TSbase instance


def confirm_imag_mode(dir_name, charge=0, mult=1, solvent=None):
    reactant_file, product_file, ts_guess_file = get_xyzs(dir_name)
    reactant, product, ts_mol = get_ade_molecules(reactant_file, product_file)

    # TODO: implement get_bond_rearrangment
    bond_rearr = get_bond_rearrangement(reactant, product)

    ade_ts = TSbase(ts_mol.atoms, reactant, product, charge=charge, mult=mult, bond_rearr=bond_rearr, solvent=solvent)
    set_hessian(ade_ts, ts_guess_file)
    ade_ts.has_correct_imag_modes()
    

def get_bond_rearrangment(reactant, product):
    pass

def get_ade_molecules(reactant_file, product_file, ts_guess_file):
    reactant = ade.Molecule(reactant_file)
    product = ade.Molecule(product_file)
    ts =  ade.Molecule(ts_guess_file)

    return reactant, product, ts


def get_xyzs(dir_name):
    reactant_file = [f for f in os.listdir(dir_name) if f == 'conformer_reactant_init'][0]
    product_file = [f for f in os.listdir(dir_name) if f == 'conformer_product_init'][0]
    ts_guess_file = [f for f in os.listdir(dir_name) if f.startswith('ts_guess_')][0]

    return reactant_file, product_file, ts_guess_file


def set_hessian(ade_ts_mol, filename):
    hessian = read_hessian(filename)
    ade_hessian = Hessian(ValueArray(hessian))
    ade_ts_mol.hessian(ade_hessian)

def read_hessian(filename):
    with open(filename, 'r') as f:
        data = f.read()
    print(' '.join(data.split()))
    numbers = list(map(float, data.split()[1:]))
    return numbers

