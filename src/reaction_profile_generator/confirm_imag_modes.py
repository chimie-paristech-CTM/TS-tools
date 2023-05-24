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

ps = Chem.SmilesParserParams()
ps.removeHs = False
bohr_ang = 0.52917721090380
workdir = 'test'

xtb = ade.methods.XTB()




# get ade reactant from xyz
# get ade product from xyz
# define instance of TSbase class
# set hessian for the TSbase instance -> see below
# apply has_correct_imag_modes method from TSbase instance


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

