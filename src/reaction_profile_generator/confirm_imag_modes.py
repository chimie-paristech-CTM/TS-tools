from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import autode as ade
import os
from functools import wraps
from scipy.spatial import distance_matrix
from scipy.spatial import distance_matrix
import re
from autode.species.species import Species
from typing import Optional
import shutil

ps = Chem.SmilesParserParams()
ps.removeHs = False
bohr_ang = 0.52917721090380

import os
import shutil

def confirm_imag_mode(dir_name):
    """
    Confirm if the provided directory represents an imaginary mode.

    Args:
        dir_name (str): The directory path.

    Returns:
        bool: True if the directory represents an imaginary mode, False otherwise.
    """
    # Obtain reactant, product, and transition state molecules
    reactant_file, product_file, ts_guess_file = get_xyzs(dir_name)
    reactant, product, ts_mol = get_ade_molecules(dir_name, reactant_file, product_file, ts_guess_file)   

    # Compute the displacement along the imaginary mode
    normal_mode, freq = read_first_normal_mode(dir_name, 'g98.out')
    f_displaced_species = displaced_species_along_mode(ts_mol, normal_mode, disp_factor=1)
    b_displaced_species = displaced_species_along_mode(reactant, normal_mode, disp_factor=-1)

    # Compute distance matrices
    f_distances = distance_matrix(f_displaced_species.coordinates, f_displaced_species.coordinates)
    b_distances = distance_matrix(b_displaced_species.coordinates, b_displaced_species.coordinates)

    # Compute delta_mode
    delta_mode = f_distances - b_distances

    # Compute delta_reaction
    reac_distances = distance_matrix(reactant.coordinates, reactant.coordinates)
    prod_distances = distance_matrix(product.coordinates, product.coordinates)
    delta_reaction = reac_distances - prod_distances

    # Get all the bonds in both reactants and products
    all_bonds = set(product.graph.edges).union(set(reactant.graph.edges))

    # Determine active bonds and extra bonds involved in the mode
    active_bonds_involved_in_mode = {}
    extra_bonds_involved_in_mode = {}

    # Identify active bonds
    active_bonds = set()
    for bond in all_bonds:
        if abs(delta_reaction[bond[0], bond[1]]) > 0.1:
            active_bonds.add(bond)

    print(active_bonds)

    # Check bond displacements and assign involvement
    for bond in all_bonds:
        if bond in active_bonds: 
            if abs(delta_mode[bond[0], bond[1]]) < 0.3:
                continue  # Small displacement, ignore
            else:
                active_bonds_involved_in_mode[bond] = delta_mode[bond[0], bond[1]]
        else:
            if abs(delta_mode[bond[0], bond[1]]) < 0.4:
                continue  # Small displacement, ignore
            else:
                extra_bonds_involved_in_mode[bond] = delta_mode[bond[0], bond[1]] 

    print(f'Main imaginary frequency: {freq} Active bonds in mode: {active_bonds_involved_in_mode};  Extra bonds in mode: {extra_bonds_involved_in_mode}')

    # Copy and rename transition state guess if conditions are met
    if len(extra_bonds_involved_in_mode) == 0 and len(active_bonds_involved_in_mode) != 0:
        path_name = '/'.join(dir_name.split('/')[:-1])
        reaction_name = dir_name.split('/')[-1]
        shutil.copy(os.path.join(dir_name, ts_guess_file), os.path.join(path_name, 'final_ts_guesses'))
        os.rename(
            os.path.join(os.path.join(path_name, 'final_ts_guesses'),  ts_guess_file), 
            os.path.join(os.path.join(path_name, 'final_ts_guesses'), f'{reaction_name}_final_ts_guess.xyz')
        )
        return True
    else:
        return False
    

def read_first_normal_mode(dir_name, filename):
    """
    Read the first normal mode from the specified file.

    Args:
        dir_name (str): The directory path.
        filename (str): The name of the file to read.

    Returns:
        numpy.ndarray: Array representing the normal mode.
        float: Frequency value.
    """
    normal_mode = []
    pattern = r'\s+(\d+)\s+\d+\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)'

    # Open the file and read its contents
    with open(os.path.join(dir_name, filename), 'r') as file:
        lines = file.readlines()
        # Iterate over the lines and find the matching pattern
        for line in lines:
            # Check if the line contains a frequency
            if 'Frequencies' in line:
                # Extract the frequency value from the line
                frequency = float(line.split('--')[1].split()[0])

                # Iterate over the lines below the frequency line
                for sub_line in lines[lines.index(line) + 7:]:
                    # Check if the line matches the pattern
                    match = re.search(pattern, sub_line)
                    if match:
                        x = float(match.group(2))
                        y = float(match.group(3))
                        z = float(match.group(4))
                        normal_mode.append([x, y, z])
                    else:
                        break
                break

    return np.array(normal_mode), frequency


def displaced_species_along_mode(
    species: Species,
    normal_mode = np.array,
    disp_factor: float = 1.0,
    max_atom_disp: float = 99.9,
) -> Optional[Species]:
    """
    Displace the geometry along a normal mode with mode number indexed from 0,
    where 0-2 are translational normal modes, 3-5 are rotational modes and 6
    is the largest magnitude imaginary mode (if present). To displace along
    the second imaginary mode we have mode_number=7

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):
        mode_number (int): Mode number to displace along

    Keyword Arguments:
        disp_factor (float): Distance to displace (default: {1.0})

        max_atom_disp (float): Maximum displacement of any atom (Å)

    Returns:
        (autode.species.Species):

    Raises:
        (autode.exceptions.CouldNotGetProperty):
    """

    coords = species.coordinates
    disp_coords = coords.copy() + disp_factor * normal_mode

    # Ensure the maximum displacement distance any single atom is below the
    # threshold (max_atom_disp), by incrementing backwards in steps of 0.05 Å,
    # for disp_factor = 1.0 Å
    for _ in range(20):

        if (
            np.max(np.linalg.norm(coords - disp_coords, axis=1))
            < max_atom_disp
        ):
            break

        disp_coords -= (disp_factor / 20) * normal_mode

    # Create a new species from the initial
    disp_species = Species(
        name=f"{species.name}_disp",
        atoms=species.atoms.copy(),
        charge=species.charge,
        mult=species.mult,
    )
    disp_species.coordinates = disp_coords

    return disp_species


def get_ade_molecules(dir_name, reactant_file, product_file, ts_guess_file):
    """
    Load the reactant, product, and transition state molecules.

    Args:
        dir_name (str): The directory path.
        reactant_file (str): The name of the reactant file.
        product_file (str): The name of the product file.
        ts_guess_file (str): The name of the transition state guess file.

    Returns:
        ade.Molecule: Reactant molecule.
        ade.Molecule: Product molecule.
        ade.Molecule: Transition state molecule.
    """
    reactant = ade.Molecule(os.path.join(dir_name, reactant_file))
    product = ade.Molecule(os.path.join(dir_name, product_file))
    ts = ade.Molecule(os.path.join(dir_name, ts_guess_file))

    return reactant, product, ts


def get_xyzs(dir_name):
    """
    Get the names of the reactant, product, and transition state guess files.

    Args:
        dir_name (str): The directory path.

    Returns:
        str: The name of the reactant file.
        str: The name of the product file.
        str: The name of the transition state guess file.
    """
    reactant_file = [f for f in os.listdir(dir_name) if f == 'conformer_reactant_init_optimised_xtb.xyz'][0]
    product_file = [f for f in os.listdir(dir_name) if f == 'conformer_product_init_optimised_xtb.xyz'][0]
    ts_guess_file = [f for f in os.listdir(dir_name) if f.startswith('ts_guess_')][0]

    return reactant_file, product_file, ts_guess_file
