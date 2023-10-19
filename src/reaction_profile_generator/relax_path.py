import numpy as np
import autode as ade
import os
import re
import subprocess

from reaction_profile_generator.confirm_imag_modes import validate_transition_state

xtb = ade.methods.XTB()
xtb.force_constant = 10


def generate_ts_guess(xyz_files, formation_constraints, breaking_constraints, charge=0):
    """ """

    final_energies_dict, sorted_energies = relax_path(xyz_files, charge)

    lowest_mode_ratio = 1 # ratio between first and second imaginary mode

    for f in [1.05, 1.005]:
        for i, energy in enumerate(sorted_energies):
            neg_freq = get_negative_frequencies(xyz_files[final_energies_dict[energy]], charge)
            if len(neg_freq) == 1:
                if validate_transition_state(xyz_files[final_energies_dict[energy]], formation_constraints, breaking_constraints, factor=f, charge=0, final=True):
                    return xyz_files[final_energies_dict[energy]]
                else:
                    continue
            elif len(neg_freq) == 0:
                continue
            else:
                current_mode_ratio = float(neg_freq[1])/float(neg_freq[0])
                if validate_transition_state(xyz_files[final_energies_dict[energy]], formation_constraints, breaking_constraints, factor=f, charge=0, final=False):
                    if current_mode_ratio < lowest_mode_ratio:
                        lowest_mode_ratio = current_mode_ratio
                        id_best_guess = final_energies_dict[energy]
                if i >= 6:
                    break
    
        if validate_transition_state(xyz_files[final_energies_dict[energy]], formation_constraints, breaking_constraints, factor=f, charge=0, final=True):
            return xyz_files[id_best_guess]


def relax_path(xyz_files, charge):
    # get projection of energy gradient in plane orthogonal to path gradient
    # move coordinates downhill in energy and repeat
    # the most rigorous explanation found is here: https://pubs.aip.org/aip/jcp/article-abstract/94/1/751/98598/Reaction-path-study-of-helix-formation-in?redirectedFrom=fulltext

    final_energies_dict = {}

    for i in range(10):
        for j, file in enumerate(xyz_files[:-1]):
            coordinates, element_types = get_coordinates(file)
            coordinates_next, _ = get_coordinates(xyz_files[j+1])

            path_gradient = coordinates - coordinates_next
            energy_gradient, energy = get_gradient(file, charge)
            projections = calculate_projections(path_gradient, energy_gradient)
            coordinates_new = coordinates - projections * 0.25
            write_xyz_file(file, element_types, coordinates_new, i)
            if i == 9:
                final_energies_dict[energy] = j

    sorted_energies = sorted(final_energies_dict.keys(), reverse=True)

    return final_energies_dict, sorted_energies


def filter_active_bonds(formed_bonds, broken_bonds):
    formed_bonds_tmp = get_bond_indices(formed_bonds)
    broken_bonds_tmp = get_bond_indices(broken_bonds)

    formed_bonds_filtered = formed_bonds_tmp.difference(broken_bonds_tmp)
    broken_bonds_filtered = broken_bonds_tmp.difference(formed_bonds_tmp)

    return formed_bonds_filtered, broken_bonds_filtered


def get_bond_indices(bond_list):
    bonds_tmp = []

    for bond in bond_list:
        i,j,_ = map(int, bond.split('-'))
        bonds_tmp.append((i,j)) 

    return set(bonds_tmp)


def write_xyz_file(filename, elements, coordinates, i):
    with open(filename, 'w') as file:
        num_atoms = len(elements)
        file.write(str(num_atoms) + "\n")
        file.write(f"This is the geometry after iteration {i}\n")
        
        for element, coordinate in zip(elements, coordinates):
            x, y, z = coordinate
            line = f"{element} {x:.6f} {y:.6f} {z:.6f}\n"
            file.write(line)

def get_coordinates(filename):
    coordinates = []
    element_types = []
    with open(filename, 'r') as file:
        # Skip the first line (number of atoms)
        next(file)
        # Skip the second line (comment line)
        next(file)
        
        for line in file:
            line = line.strip()
            if line:
                atom_data = line.split()
                element_types.append(atom_data[0])
                x, y, z = map(float, atom_data[1:4])
                coordinates.append([x, y, z])
    
    return np.array(coordinates), element_types


def calculate_projections(vectors1, vectors2):
    # calculate the projection of vectors2 in the planes perpendicular to vectors1
    projections = []
    for i in range(len(vectors1)):
        projections.append(vectors2[i] - np.dot(vectors2[i], vectors1[i]) / 
                           np.dot(vectors1[i], vectors1[i]) * vectors1[i])

    return np.array(projections)


def get_gradient(filename, charge):
    with open('gradient.out', 'w') as out:
        process = subprocess.Popen(f'xtb {filename} --charge {charge} --grad'.split(), 
                                   stderr=subprocess.DEVNULL, stdout=out)
        process.wait()
    
    gradient = extract_gradient_from_file('gradient')

    return gradient

def extract_gradient_from_file(file_path):
    gradient_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if 'cycle' in line and 'SCF energy' in line:
                words = line.split()
                energy = float(words[6])
            elif len(line.split()) == 3:  # Check if line contains three numbers
                gradient_lines.append(line)

    gradient = [[float(num) for num in line.split()] for line in gradient_lines]
    
    return np.array(gradient), energy


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
            