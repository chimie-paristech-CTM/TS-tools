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


def refine_ts(charge=0):
    # get bond length reactant and product
    # compare active bond lengths in TS structure with optimal bond lengths R and P
    # take most activated bond and perform relaxed scan starting from guess geometry 
    # do freq calculation of the approximate maximum along the scan direction

    # Obtain reactant, product, and transition state molecules
    reactant_file, product_file, ts_guess_file = get_xyzs()
    reactant, product, ts_mol = get_ade_molecules(reactant_file, product_file, ts_guess_file, charge)   

    # Compute interatom distances for reactant, product and TS
    reac_distances = distance_matrix(reactant.coordinates, reactant.coordinates)
    prod_distances = distance_matrix(product.coordinates, product.coordinates)
    delta_reaction = reac_distances - prod_distances
    ts_distances = distance_matrix(ts_mol.coordinates, ts_mol.coordinates)

    # Get all the bonds in both reactants and products
    all_bonds = set(product.graph.edges).union(set(reactant.graph.edges))

    displacements = {}
    for bond in all_bonds:
        if abs(delta_reaction[bond[0], bond[1]]) > 0.5:
            displacement = min(
                ts_distances[bond[0], bond[1]] - min(reac_distances[bond[0], bond[1]], prod_distances[bond[0], bond[1]]),
                max(reac_distances[bond[0], bond[1]], prod_distances[bond[0], bond[1]]) - ts_distances[bond[0], bond[1]]
            )
            #print(bond, delta_reaction[bond[0], bond[1]], displacement)
            displacements[bond] = displacement

    sorted_bonds = sorted(displacements.keys(), key=lambda k: displacements[k], reverse=True)

    for bond in sorted_bonds:
        print(bond, reac_distances[bond[0], bond[1]], prod_distances[bond[0], bond[1]])
        energies = []
        xyz_file_names = []

        print(np.linspace(max(min(
            reac_distances[bond[0], bond[1]], prod_distances[bond[0], bond[1]]), 1), \
            min(max(reac_distances[bond[0], bond[1]], prod_distances[bond[0], bond[1]]), 5), 10))

        for distance in np.linspace(max(min(
            reac_distances[bond[0], bond[1]], prod_distances[bond[0], bond[1]]), 1), \
            min(max(reac_distances[bond[0], bond[1]], prod_distances[bond[0], bond[1]]), 5), 10): #range(-5,5):
            ts_tmp = ade.Molecule(ts_guess_file, charge=charge)
            ts_tmp.constraints.update({bond: distance})
            ts_tmp.name = f'ts_scan_{distance}'
            ts_tmp.optimise(xtb)
            energies.append(ts_tmp.energy)
            xyz_file_names.append(ts_tmp.name)

        print(energies)
        # determine the maximum -> TODO: FIT PARABOLA
        for i in range(1, len(energies)-1):
            if energies[i] > energies[i+1] and energies[i] > energies[i-1]:
                get_negative_frequencies(f'{xyz_file_names[i]}_optimised_xtb.xyz', charge)
                return f'{xyz_file_names[i]}_optimised_xtb.xyz'
            else:
                continue
    
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
            