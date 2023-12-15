import os
from rdkit import Chem
import subprocess
import shutil

ps = Chem.SmilesParserParams()
ps.removeHs = False


def xyz_to_gaussian_input(xyz_file, output_file, method='UB3LYP', basis_set='6-31G(d,p)', 
                          extra_commands='opt=(calcfc,ts, noeigen) freq=noraman', charge=0, multiplicity=1):
    """
    Convert an XYZ file to Gaussian 16 input file format.

    Args:
        xyz_file (str): Path to the XYZ file.
        output_file (str): Path to the output Gaussian input file to be created.
        method (str, optional): The method to be used in the Gaussian calculation. Default is 'B3LYP'.
        basis_set (str, optional): The basis set to be used in the Gaussian calculation. Default is '6-31G(d)'.
    """
    filename = xyz_file.split('/')[-1].split('.xyz')[0]

    with open(xyz_file, 'r') as xyz:
        atom_lines = xyz.readlines()[2:]  # Skip the first two lines (number of atoms and comment)

    with open(output_file, 'w') as gaussian_input:
        # Write the route section
        if 'external' in method:
            gaussian_input.write(f'%Chk={filename}.chk\n# {method} {extra_commands}')
        else:
            gaussian_input.write(f'%Chk={filename}.chk\n%nproc=8\n%Mem=16GB\n# {method}/{basis_set} {extra_commands}')
        
        # Write the title section
        gaussian_input.write('\n\nTitle\n\n')

        # Write the charge and multiplicity section
        gaussian_input.write(f'{charge} {multiplicity}\n')

        # Write the Cartesian coordinates section
        for line in atom_lines:
            atom_info = line.split()
            element = atom_info[0]
            x, y, z = atom_info[1:4]
            gaussian_input.write(f'{element} {x} {y} {z}\n')

        # Write the blank line and the end of the input file
        gaussian_input.write('\n')

    print(f'Gaussian input file "{output_file}" has been created.')


def write_xyz_file_from_ade_atoms(atoms, filename):
    """
    Write an XYZ file from the ADE atoms object.

    Args:
        atoms: The ADE atoms object.
        filename: The name of the XYZ file to write.
    """
    with open(filename, 'w') as f:
        f.write(str(len(atoms)) + '\n')
        f.write('Generated by write_xyz_file()\n')
        for atom in atoms:
            f.write(f'{atom.atomic_symbol} {atom.coord[0]:.6f} {atom.coord[1]:.6f} {atom.coord[2]:.6f}\n')


def write_final_geometry_to_xyz(log_file_path):
    final_geometry = []
    reading_geometry = False
    after_transition_state_opt = False

    xyz_file_path = os.path.splitext(log_file_path)[0] + ".xyz"

    try:
        with open(log_file_path, 'r') as log_file:
            for line in log_file:
                if 'Stationary point found' in line:
                    after_transition_state_opt = True
                if after_transition_state_opt:
                    if 'Standard orientation' in line:
                        reading_geometry = True
                        final_geometry = []
                    elif reading_geometry:
                        # Lines with atomic coordinates are indented
                        columns = line.split()
                        if len(columns) == 6:
                            try:
                                atom_info = {
                                    'atom': int(columns[0]),
                                    'symbol': int(columns[1]),
                                    'x': float(columns[3]),
                                    'y': float(columns[4]),
                                    'z': float(columns[5])
                                }
                                final_geometry.append(atom_info)
                            except:
                                continue
                        else:
                            if len(final_geometry) != 0 and '-----------------------------' in line:
                                break

        if len(final_geometry) != 0:
            with open(xyz_file_path, 'w') as xyz_file:
                num_atoms = len(final_geometry)
                xyz_file.write(f"{num_atoms}\n")
                xyz_file.write("Final geometry extracted from Gaussian log file\n")
                for atom_info in final_geometry:
                    xyz_file.write(f"{atom_info['symbol']} {atom_info['x']:.6f} {atom_info['y']:.6f} {atom_info['z']:.6f}\n")

    except FileNotFoundError:
        print(f"File not found: {log_file_path}")
    
    return xyz_file_path


def run_g16_ts_optimization(file_path):
    # Run Gaussian 16 using nohup and redirect stderr to /dev/null
    log_file = os.path.splitext(file_path)[0] + ".log"
    out_file = os.path.splitext(file_path)[0] + ".out"
                
    with open(out_file, 'w') as out:
        subprocess.run(
            f"g16 < {file_path} > {log_file}",
            shell=True,
            stdout=out,
            stderr=subprocess.DEVNULL,
        ) 
    
    return log_file

def run_irc(input_file):
    out_file = f'{input_file[:-4]}.out'
    log_file = f'{input_file[:-4]}.log'

    with open(out_file, 'w') as out:
        subprocess.run(
            f"g16 < {input_file} >> {log_file}",
            shell=True,
            stdout=out,
            stderr=subprocess.DEVNULL,
        )

def copy_final_outputs(dir_name):
    os.makedirs('final_outputs', exist_ok=True)
    for reaction_dir in os.listdir(dir_name):
        try:
            final_ts_guess_dir = os.path.join(os.path.join(dir_name, reaction_dir), 'final_ts_guess')
            if len(os.listdir(final_ts_guess_dir)) != 0:
                shutil.copytree(final_ts_guess_dir, os.path.join('final_outputs', f'final_outputs_{reaction_dir}'))
                shutil.copy(os.path.join(os.path.join(dir_name, reaction_dir), 'rp_geometries/reactants_geometry.xyz'),
                        os.path.join('final_outputs', f'final_outputs_{reaction_dir}/'))
                shutil.copy(os.path.join(os.path.join(dir_name, reaction_dir), 'rp_geometries/products_geometry.xyz'),
                        os.path.join('final_outputs', f'final_outputs_{reaction_dir}/'))
        except:
            continue


def remove_files_in_directory(directory):
    try:
        # List all items in the directory
        items = os.listdir(directory)

        # Iterate over each item and remove only files
        for item_name in items:
            item_path = os.path.join(directory, item_name)

            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                continue

    except Exception as e:
        print(f"Error during file removal: {e}")


if __name__ == '__main__':
   write_final_geometry_to_xyz('logs/ts_guess_2.log')