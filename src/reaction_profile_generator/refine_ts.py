from rdkit import Chem
import numpy as np
import autode as ade
import os
from scipy.spatial import distance_matrix
import re
from autode.species.species import Species
from typing import Optional
import shutil
import subprocess

from reaction_profile_generator.utils import write_xyz_file_from_ade_atoms

xtb = ade.methods.XTB()
xtb.force_constant = 10


def refine_ts(reaction_path_gradient, charge=0):
    # get plane orthogonal to path gradient
    # compute actual gradient of the molecule (by running --grad; cf. https://xtb-docs.readthedocs.io/en/latest/commandline.html)
    # project the gradient in the perpendicular plane and do again
    # the most rigorous explanation found is here: https://pubs.aip.org/aip/jcp/article-abstract/94/1/751/98598/Reaction-path-study-of-helix-formation-in?redirectedFrom=fulltext

    _, _, ts_guess_file = get_xyzs()
    
    return None


def get_ade_molecules(reactant_file, product_file, ts_guess_file, charge):
    """
    Load the reactant, product, and transition state molecules.

    Args:
        reactant_file (str): The name of the reactant file.
        product_file (str): The name of the product file.
        ts_guess_file (str): The name of the transition state guess file.

    Returns:
        ade.Molecule: Reactant molecule.
        ade.Molecule: Product molecule.
        ade.Molecule: Transition state molecule.
    """
    reactant = ade.Molecule(reactant_file, charge=charge)
    product = ade.Molecule(product_file, charge=charge)
    ts = ade.Molecule(ts_guess_file, charge=charge)

    return reactant, product, ts


def get_xyzs():
    """
    Get the names of the reactant, product, and transition state guess files.

    Returns:
        str: The name of the reactant file.
        str: The name of the product file.
        str: The name of the transition state guess file.
    """
    reactant_file = [f for f in os.listdir() if f == 'conformer_reactant_init_optimised_xtb.xyz'][0]
    product_file = [f for f in os.listdir() if f == 'conformer_product_init_optimised_xtb.xyz'][0]
    ts_guess_file = [f for f in os.listdir() if f.startswith('ts_guess_')][0]

    return reactant_file, product_file, ts_guess_file


# deduplicate
def get_negative_frequencies(filename, charge):
    """
    Executes an external program to calculate the negative frequencies for a given file.

    Args:
        filename (str): The name of the file to be processed.
        charge (int): The charge value for the calculation.

    Returns:
        list: A list of negative frequencies.
    """
    with open('hess.out', 'w') as out:
        process = subprocess.Popen(f'xtb {filename} --charge {charge} --hess'.split(), 
                                   stderr=subprocess.DEVNULL, stdout=out)
        process.wait()
    
    neg_freq = read_negative_frequencies('g98.out')
    return neg_freq


def read_negative_frequencies(filename):
    """
    Read the negative frequencies from a file.

    Args:
        filename: The name of the file.

    Returns:
        list: The list of negative frequencies.
    """
    with open(filename, 'r') as file:
        for line in file:
            if line.strip().startswith('Frequencies --'):
                frequencies = line.strip().split()[2:]
                negative_frequencies = [freq for freq in frequencies if float(freq) < 0]
                return negative_frequencies
            