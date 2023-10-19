from rdkit import Chem
import numpy as np
import autode as ade
import os
from scipy.spatial import distance_matrix
import re
from autode.species.species import Species
from typing import Optional
import shutil

from reaction_profile_generator.utils import write_xyz_file_from_ade_atoms

xtb = ade.methods.XTB()


def validate_transition_state(ts_file, charge=0, final=False):
    """
    Validates a transition state by performing various checks on the provided parameters.

    Args:
        ts_file (str): The name of the TS file.
        charge (int): The charge of the reacting system. Defaults to 0.

    Returns:
        bool: True if the transition state is valid, False otherwise.
    """

    freq, active_bonds_involved_in_mode, extra_bonds_involved_in_mode, \
    active_bonds_forming, active_bonds_breaking, reac_distances, prod_distances = confirm_imag_mode(ts_file, charge)

    ts_mol = ade.Molecule(ts_file)
    ts_distances = distance_matrix(ts_mol.coordinates, ts_mol.coordinates)

    active_formed_bonds_involved_in_mode = [
        active_bonds_involved_in_mode[bond]
        for bond in set(active_bonds_involved_in_mode.keys()).intersection(active_bonds_forming)
    ]
    active_broken_bonds_involved_in_mode = [
        active_bonds_involved_in_mode[bond]
        for bond in set(active_bonds_involved_in_mode.keys()).intersection(active_bonds_breaking)
    ]

    print(f'Main imaginary frequency: {freq} Active bonds in mode: {active_bonds_involved_in_mode};  Extra bonds in mode: {extra_bonds_involved_in_mode}')

    # Check that at least one active bond is getting displaced in mode,
    # check that all broken and formed bonds in the active mode move in the same direction,
    # and that the bond lengths are intermediate between reactants and products.
    #bond_lengths_intermediate = check_if_bond_lengths_intermediate(
    #    ts_distances, reac_distances, prod_distances, active_bonds_forming, active_bonds_breaking
    #)
    bond_lengths_intermediate = check_if_bond_lengths_intermediate(ts_distances, reac_distances, prod_distances, 
                                                                    active_bonds_forming, active_bonds_breaking)
    # TODO: Fix this!

    if len(extra_bonds_involved_in_mode) == 0 and len(active_bonds_involved_in_mode) != 0 \
    and check_same_sign(active_formed_bonds_involved_in_mode) \
    and check_same_sign(active_broken_bonds_involved_in_mode) and bond_lengths_intermediate and freq < -50:
        if final:
            move_final_guess_xyz(ts_file)
        return True
    else:
        return False

# TODO: you may want to express this in terms of covalent radii (see Jensen)
def check_if_bond_lengths_intermediate(ts_distances, reac_distances, prod_distances, active_bonds_forming, active_bonds_breaking):
    """
    Checks if the bond lengths in the transition state are intermediate between reactants and products.

    Args:
        ts_distances (numpy.ndarray): Distance matrix of the transition state.
        reac_distances (numpy.ndarray): Distance matrix of the reactants.
        prod_distances (numpy.ndarray): Distance matrix of the products.
        active_bonds_forming (list): List of active bonds involved in bond formation.
        active_bonds_breaking (list): List of active bonds involved in bond breaking.

    Returns:
        bool: True if the bond lengths are intermediate, False otherwise.
    """

    for active_bond in active_bonds_forming:
        if ts_distances[active_bond[0], active_bond[1]] <= prod_distances[active_bond[0], active_bond[1]] * 1.03:
            print(active_bond, ts_distances[active_bond[0], active_bond[1]], reac_distances[active_bond[0], active_bond[1]])
            return False
        else:
            continue
    
    for active_bond in active_bonds_breaking:
        if ts_distances[active_bond[0], active_bond[1]] <= reac_distances[active_bond[0], active_bond[1]] * 1.03:
            print(active_bond, ts_distances[active_bond[0], active_bond[1]], reac_distances[active_bond[0], active_bond[1]])
            return False
        else:
            continue
    
    return True


def check_same_sign(mode_list):
    """
    Checks if all numbers in the given list have the same sign.

    Args:
        mode_list (list): List of numbers.

    Returns:
        bool: True if all numbers have the same sign, False otherwise.
    """

    first_sign = 0

    for num in mode_list:
        if num > 0:
            sign = 1
        else:
            sign = -1

        if first_sign == 0:
            first_sign = sign
        elif sign != 0 and sign != first_sign:
            return False
        
    return True


def move_final_guess_xyz(ts_guess_file):
    """
    Moves the final transition state guess XYZ file to a designated folder and renames it.

    Args:
        ts_guess_file (str): Path to the transition state guess XYZ file.

    Returns:
        None
    """

    path_name = '/'.join(os.getcwd().split('/')[:-1])
    reaction_name = os.getcwd().split('/')[-1]
    shutil.copy(ts_guess_file, os.path.join(path_name, 'final_ts_guesses'))
    os.rename(
        os.path.join(os.path.join(path_name, 'final_ts_guesses'), ts_guess_file.split('/')[-1]), 
        os.path.join(os.path.join(path_name, 'final_ts_guesses'), f'{reaction_name}_final_ts_guess.xyz')
    )


def confirm_imag_mode(ts_file, charge):
    """
    Confirm if the provided directory represents an imaginary mode.

    Args:
        dir_name (str): The directory path.

    Returns:
        bool: True if the directory represents an imaginary mode, False otherwise.
    """
    # Obtain reactant, product, and transition state molecules
    reactant_file, product_file = get_xyzs()
    reactant, product, ts_mol = get_ade_molecules(reactant_file, product_file, ts_file, charge)   

    # Compute the displacement along the imaginary mode
    normal_mode, freq = read_first_normal_mode('g98.out')
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
    active_bonds_forming = set()
    active_bonds_breaking = set()
    active_bonds = set()
    for bond in all_bonds:
        if delta_reaction[bond[0], bond[1]] > 0.1:
            active_bonds_forming.add(bond)
            active_bonds.add(bond)
        if delta_reaction[bond[0], bond[1]] < -0.1:
            active_bonds_breaking.add(bond)
            active_bonds.add(bond)

    # Check bond displacements and assign involvement
    for bond in all_bonds:
        if bond in active_bonds: 
            if abs(delta_mode[bond[0], bond[1]]) < 0.3:
                continue  # Small displacement, ignore
            else:
                active_bonds_involved_in_mode[bond] = delta_mode[bond[0], bond[1]]
        else:
            if abs(delta_mode[bond[0], bond[1]]) < 0.3:
                continue  # Small displacement, ignore
            else:
                extra_bonds_involved_in_mode[bond] = delta_mode[bond[0], bond[1]] 

    return freq, active_bonds_involved_in_mode, extra_bonds_involved_in_mode, active_bonds_forming, active_bonds_breaking, reac_distances, prod_distances


def read_first_normal_mode(filename):
    """
    Read the first normal mode from the specified file.

    Args:
        filename (str): The name of the file to read.

    Returns:
        numpy.ndarray: Array representing the normal mode.
        float: Frequency value.
    """
    normal_mode = []
    pattern = r'\s+(\d+)\s+\d+\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)'

    # Open the file and read its contents
    with open(filename, 'r') as file:
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

    return reactant_file, product_file
