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
from autode.transition_states.base import TSbase
from autode.bond_rearrangement import prune_small_ring_rearrs, strip_equiv_bond_rearrs, add_bond_rearrangment
import math
from autode.mol_graphs import (
    get_bond_type_list,
    get_fbonds,
    is_isomorphic,
    find_cycles,
)
import itertools

ps = Chem.SmilesParserParams()
ps.removeHs = False
bohr_ang = 0.52917721090380

xtb = ade.methods.XTB()


# get ade reactant from xyz
# get ade product from xyz
# define instance of TSbase class
# set hessian for the TSbase instance -> see below
# apply has_correct_imag_modes method from TSbase instance


# use displaced_species_along_mode to generate geometries for tentative reactants/products, save as xyz, generate ade mols and then compare to actual react + prod

def confirm_imag_mode(dir_name, charge=0, mult=1, solvent=None):
    reactant_file, product_file, ts_guess_file = get_xyzs(dir_name)
    reactant, product, ts_mol = get_ade_molecules(dir_name, reactant_file, product_file, ts_guess_file)

    # TODO: implement get_bond_rearrangment
    bond_rearr = get_bond_rearrangs(reactant, product)

    print(bond_rearr)

    ade_ts = TSbase(ts_mol.atoms, reactant, product, charge=charge, mult=mult, bond_rearr=bond_rearr, solvent_name=solvent)
    ade_ts = set_hessian(dir_name, ade_ts, 'hessian')
    ade_ts.has_correct_imag_mode()
    print(ade_ts.has_correct_imag_mode())
    

def get_ade_molecules(dir_name, reactant_file, product_file, ts_guess_file):
    reactant = ade.Molecule(os.path.join(dir_name, reactant_file))
    product = ade.Molecule(os.path.join(dir_name, product_file))
    ts =  ade.Molecule(os.path.join(dir_name, ts_guess_file))

    return reactant, product, ts


def get_xyzs(dir_name):
    reactant_file = [f for f in os.listdir(dir_name) if f == 'conformer_reactant_init.xyz'][0]
    product_file = [f for f in os.listdir(dir_name) if f == 'conformer_product_init.xyz'][0]
    ts_guess_file = [f for f in os.listdir(dir_name) if f.startswith('ts_guess_')][0]

    return reactant_file, product_file, ts_guess_file


def set_hessian(dir_name, ade_ts, filename):
    hessian = convert_to_2d_array(read_hessian(os.path.join(dir_name, filename)))
    print(hessian)
    ade_hessian = Hessian(ValueArray(hessian))
    print(ade_ts, ade_hessian)
    ade_ts.hessian = ade_hessian

    return ade_ts


def read_hessian(filename):
    with open(filename, 'r') as f:
        data = f.read()
    numbers = list(map(float, data.split()[1:]))

    return numbers


def convert_to_2d_array(arr):
    size = int(math.sqrt(len(arr)))  # Determine the size of the square array
    if size * size != len(arr):
        raise ValueError("Input array length is not compatible with a square array.")

    return np.array([arr[i:i+size] for i in range(0, len(arr), size)])

# TODO: this is trash!
def get_bond_rearrangs(reactant, product):
    """For a reactant and product (mol_complex) find the set of breaking and
    forming bonds that will turn reactants into products. This works by
    determining the types of bonds that have been made/broken (i.e CH) and
    then only considering rearrangements involving those bonds.

    ---------------------------------------------------------------------------
    Arguments:
        reactant (autode.species.ReactantComplex):

        product (autode.species.ProductComplex):

    Returns:
        (list(autode.bond_rearrangements.BondRearrangement)):
    """

    possible_brs = []

    reac_bond_dict = get_bond_type_list(reactant.graph)
    prod_bond_dict = get_bond_type_list(product.graph)

    # list of bonds where this type of bond (e.g C-H) has less bonds in
    # products than reactants
    all_possible_bbonds = []

    # list of bonds that can be formed of this bond type. This is only used
    # if there is only one type of bbond, so can be overwritten for each new
    # type of bbond
    bbond_atom_type_fbonds = None

    # list of bonds where this type of bond (e.g C-H) has more bonds in
    #  products than reactants
    all_possible_fbonds = []

    # list of bonds that can be broken of this bond type. This is only used
    # if there is only one type of fbond, so can be overwritten for each new
    # type of fbond
    fbond_atom_type_bbonds = None

    # list of bonds where this type of bond (e.g C-H) has the same number of
    # bonds in products and reactants
    possible_bbond_and_fbonds = []

    for reac_key, reac_bonds in reac_bond_dict.items():
        prod_bonds = prod_bond_dict[reac_key]
        possible_fbonds = get_fbonds(reactant.graph, reac_key)
        if len(prod_bonds) < len(reac_bonds):
            all_possible_bbonds.append(reac_bonds)
            bbond_atom_type_fbonds = possible_fbonds
        elif len(prod_bonds) > len(reac_bonds):
            all_possible_fbonds.append(possible_fbonds)
            fbond_atom_type_bbonds = reac_bonds
        else:
            if len(reac_bonds) != 0:
                possible_bbond_and_fbonds.append([reac_bonds, possible_fbonds])

    # The change in the number of bonds is > 0 as in the reaction
    # initialisation reacs/prods are swapped if this is < 0
    delta_n_bonds = (
        reactant.graph.number_of_edges() - product.graph.number_of_edges()
    )

    funcs = [get_fbonds_bbonds]

    for func in funcs:
        possible_brs = func(
            reactant,
            product,
            possible_brs,
            all_possible_bbonds,
            all_possible_fbonds,
            possible_bbond_and_fbonds,
            bbond_atom_type_fbonds,
            fbond_atom_type_bbonds,
            delta_n_bonds
        )

        if len(possible_brs) > 0:
            n_bond_rearrangs = len(possible_brs)
            if n_bond_rearrangs > 1:
                possible_brs = strip_equiv_bond_rearrs(possible_brs, reactant)
                prune_small_ring_rearrs(possible_brs, reactant)

            return possible_brs

    return None

def get_fbonds_bbonds(
    reac,
    prod,
    possible_brs,
    all_possible_bbonds,
    all_possible_fbonds,
    possible_bbond_and_fbonds,
    bbond_atom_type_fbonds,
    fbond_atom_type_bbonds,
    delta_n_bonds,
):
    print(delta_n_bonds)
    raise KeyError
    if len(all_possible_bbonds) == 1:
        # Break two bonds of the same type
        for bbond1, bbond2 in itertools.combinations(
            all_possible_bbonds[0], delta_n_bonds
        ):
            possible_brs = add_bond_rearrangment(
                possible_brs, reac, prod, fbonds=[], bbonds=[bbond1, bbond2]
            )

    elif len(all_possible_bbonds) == 2:
        # Break two bonds of different types
        for bbond1, bbond2 in itertools.product(
            all_possible_bbonds[0], all_possible_bbonds[1]
        ):

            possible_brs = add_bond_rearrangment(
                possible_brs, reac, prod, fbonds=[], bbonds=[bbond1, bbond2]
            )

    return possible_brs