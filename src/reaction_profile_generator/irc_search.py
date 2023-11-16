import re
import autode as ade
import os

atomic_number_to_symbol = {
    1: 'H',  2: 'He',  3: 'Li',  4: 'Be',  5: 'B',
    6: 'C',  7: 'N',   8: 'O',   9: 'F',  10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S',  17: 'Cl', 18: 'Ar', 19: 'K',  20: 'Ca',
    21: 'Sc', 22: 'Ti', 23: 'V',  24: 'Cr', 25: 'Mn',
    26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
    31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br',
    36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y',  40: 'Zr',
    41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh',
    46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I',  54: 'Xe', 55: 'Cs',
    56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
    61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb',
    66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
    71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W',  75: 'Re',
    76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
    81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
    86: 'Rn'
}


def extract_irc_geometries(log_path):
    geometry_start_pattern = re.compile(r'^\s*Cartesian Coordinates \(Ang\):\s*$')
    geometry_end_pattern = re.compile(r'^\s*CHANGE IN THE REACTION COORDINATE =\s*([+-]?\d*\.\d+)\s*$')
    reverse_path_pattern = re.compile(r'^\s*Beginning calculation of the REVERSE path\.\s*$')

    reverse_path_started = False

    with open(log_path, 'r') as log_file:
        lines = log_file.readlines()

        for i, line in enumerate(lines):
            if reverse_path_pattern.match(line):
                reverse_path_started = True
            if not reverse_path_started and geometry_start_pattern.match(line.strip()):
                geometry_block_forward_start_line = i + 5
            if not reverse_path_started and geometry_end_pattern.search(line):
                geometry_block_forward_end_line = i - 1

            if reverse_path_started and geometry_start_pattern.match(line.strip()):
                geometry_block_reverse_start_line = i + 5
            if reverse_path_started and geometry_end_pattern.search(line):
                geometry_block_reverse_end_line = i - 1

    geometry_block_forward = lines[geometry_block_forward_start_line:geometry_block_forward_end_line]
    geometry_block_reverse = lines[geometry_block_reverse_start_line:geometry_block_reverse_end_line]

    write_geometry_block_to_xyz(geometry_block_forward, f'{log_path[:-4]}_forward.xyz', True)
    write_geometry_block_to_xyz(geometry_block_reverse, f'{log_path[:-4]}_reverse.xyz', True)


def write_geometry_block_to_xyz(geometry_block, output_xyz_path, irc=False):
    with open(output_xyz_path, 'w') as xyz_file:
        # Write the number of atoms
        xyz_file.write(str(len(geometry_block)) + '\n\n')
        # Write the atomic coordinates
        for line in geometry_block:
            split_line = line.split()
            if irc:
                xyz_file.write(f'{atomic_number_to_symbol[int(split_line[1])]} {float(split_line[2])} {float(split_line[3])} {float(split_line[4])}\n')
            else:
                xyz_file.write(f'{atomic_number_to_symbol[int(split_line[1])]} {float(split_line[3])} {float(split_line[4])} {float(split_line[5])}\n')


    # iterate through lines, each time update block until you reach backward direction, then do the same and return both xyz files
    

def extract_transition_state_geometry(log_path, output_xyz_path):
    # Define regular expressions to identify relevant lines
    geometry_start_pattern = re.compile(r'^\s*Standard orientation:.*')
    geometry_end_pattern = re.compile(r'^\s*-*\n')
    transition_state_pattern = re.compile(r'^\s*-- Stationary point found\.$')

    # Set boolean flag
    ts_found = False
    geometry_block_start_line = 0
    geometry_block_end_line = 0

    # Read the Gaussian16 log file
    with open(log_path, 'r') as log_file:
        lines = log_file.readlines()

        # Iterate through the lines in the log file
        for i, line in enumerate(lines):
            # Break if transition state is found
            if transition_state_pattern.match(line):
                ts_found = True

            if ts_found and geometry_start_pattern.match(line):
                geometry_block_start_line = i + 5
            
            if ts_found and geometry_block_start_line != 0 and i > geometry_block_start_line + 5 and geometry_end_pattern.match(line):
                geometry_block_end_line = i
                break

    geometry_block = lines[geometry_block_start_line: geometry_block_end_line]

    # Write the final geometry to an XYZ file
    write_geometry_block_to_xyz(geometry_block, f'{log_path[:-4]}.xyz')

    print(f'Transition state geometry extracted and saved to {output_xyz_path}.')


def compare_molecules_irc(forward_xyz, reverse_xyz, reactant_xyz, product_xyz, charge=0):
    forward_mol = ade.Molecule(forward_xyz, name='forward', charge=charge)
    reverse_mol = ade.Molecule(reverse_xyz, name='reverse', charge=charge)
    reactant_mol = ade.Molecule(reactant_xyz, name='reactant', charge=charge)
    product_mol = ade.Molecule(product_xyz, name='product', charge=charge)
    if set(forward_mol.graph.edges) == set(reactant_mol.graph.edges) and set(reverse_mol.graph.edges) == set(product_mol.graph.edges):
        return True
    elif set(forward_mol.graph.edges) == set(product_mol.graph.edges) and set(reverse_mol.graph.edges) == set(reactant_mol.graph.edges): 
        return True
    else:
        return False


def generate_gaussian_irc_input(xyz_file, output_prefix='irc_calc', method='B3LYP/6-31G*', 
                                mem='2GB', proc=2):
    # Read the XYZ file
    with open(xyz_file, 'r') as xyz:
        lines = xyz.readlines()

    # Extract the atomic coordinates
    atom_coords = lines[2:]

    # Define the Gaussian input file content
    if 'external' not in method:
        input_content = f'%Chk={xyz_file.split("/")[-1][:-4]}.chk\n%NProc={proc}\n%Mem={mem}\n#p IRC(calcfc, maxpoint=30, stepsize=10) {method}' \
            f'\n\nIRC Calculation\n\n0 1\n{"".join(atom_coords)}\n\n' 
    else:
        input_content = f'%Chk={xyz_file.split("/")[-1][:-4]}.chk\n#p IRC(calcfc, maxpoint=30, stepsize=10) {method}\n\n' \
            f'IRC Calculation\n\n0 1\n{"".join(atom_coords)}\n\n' 

    # Write the input content to a Gaussian input file
    input_filename = os.path.join(os.path.dirname(xyz_file), f'{output_prefix}.com')
    print(input_filename)
    with open(input_filename, 'w') as input_file:
        input_file.write(input_content)

    return input_filename

if __name__ == '__main__':
    #path = '/Users/thijsstuyver/Desktop/reaction_profile_generator/benchmarking_2.0_200/final_ts_guesses/reaction_R100_final_ts_guess_0.xyz'
    path = '/Users/thijsstuyver/Desktop/reaction_profile_generator/lol.log'
    extract_transition_state_geometry(path, f'{path[:-4]}.xyz')
    generate_gaussian_irc_input(f'{path[:-4]}.xyz', method='external="/home/thijs/Jensen_xtb_gaussian/profiles_test/extra/xtb_external.py"')
    extract_irc_geometries('/Users/thijsstuyver/Desktop/reaction_profile_generator/test_irc.log')
    reaction_correct = compare_molecules_irc('/Users/thijsstuyver/Desktop/reaction_profile_generator/test_irc_forward.xyz', 
                          '/Users/thijsstuyver/Desktop/reaction_profile_generator/test_irc_reverse.xyz',
                          '/Users/thijsstuyver/Desktop/reaction_profile_generator/reactants.xyz', 
                          '/Users/thijsstuyver/Desktop/reaction_profile_generator/products.xyz')
    print(reaction_correct)
