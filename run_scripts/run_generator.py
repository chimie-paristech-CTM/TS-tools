import time
import os
import shutil
import subprocess
import multiprocessing
from functools import partial

from reaction_profile_generator.get_path import determine_ts_guess_from_path
from reaction_profile_generator.confirm_imag_modes import validate_ts_guess
from reaction_profile_generator.utils import work_in, xyz_to_gaussian_input

# TODO: sort out issue with workdir!
workdir = ['test']

atom_dict = {1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}


def change_workdir(new_name):
    workdir[0] = new_name


def get_smiles_strings(filename):
    ''' a function that opens a file, reads in every line as a reaction smiles and returns them as a list. '''
    with open(filename, 'r') as f:
        lines = f.readlines()
        smiles_strings = [line.rstrip().split() for line in lines]
    return smiles_strings


def get_smiles_strings_alt():
    return [#['R73','[C:1](=[C:2]([C:3](=[C:4](\[H:14])[H:15])\[H:16])/[H:7])(\[H:8])[H:9].[C:5](=[C:6](/[H:12])[H:13])(\[H:10])[H:11]>>[C:1]1([H:8])([H:9])[C:2]([H:7])=[C:3]([H:16])[C:4]([H:14])([H:15])[C:6]([H:12])([H:13])[C:5]1([H:10])[H:11]'],
        ['R74','[C@:1]1([C:5]([H:11])([H:12])[H:13])([H:7])[C:2]([H:8])=[C:3]([H:9])[C@@:4]1([C:6]([H:14])([H:15])[H:16])[H:10]>>[C:1](=[C:2]([C:3](=[C:4](/[C:6]([H:14])([H:15])[H:16])[H:10])/[H:9])\[H:8])(\[C:5]([H:11])([H:12])[H:13])[H:7]']]


@work_in(workdir)
def obtain_ts_guess(reaction_smiles, reactive_complex_factor=1.8, bond_length_factor=1.05, disp_cut_off=0.7, freq_cut_off=150):
    ''' a function that splits up a reaction smiles in reactant and product, and then calls the function get_path with these as parameters. '''
    reactant_smiles, product_smiles = reaction_smiles.split('>>')
    ts_guess = determine_ts_guess_from_path(reactant_smiles, product_smiles, reactive_complex_factor=reactive_complex_factor, 
                                 bond_length_factor=bond_length_factor, disp_cut_off=disp_cut_off, freq_cut_off=freq_cut_off) #, solvent='water')

    return ts_guess

def setup_dirs(target_dir):
    if target_dir in os.listdir():
        shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    os.mkdir(f'{target_dir}/final_ts_guesses')
    os.mkdir(f'{target_dir}/g16_input_files')

    return target_dir


def print_statistics_guess_generation(successful_reactions, start_time):
    end_time = time.time()
    print(f'Successful reactions: {successful_reactions}')
    print(f'Number of successful reactions: {len(successful_reactions)}')
    print(f'Time taken: {end_time - start_time}')


def generate_gaussian_inputs(target_dir, method='external="../xtb_external.py"', basis_set='', 
                             extra_commands='opt=(ts, calcall, noeigen, nomicro)'):
    for file in os.listdir(f'{target_dir}/final_ts_guesses'):
        xyz_file = os.path.join(f'{target_dir}/final_ts_guesses', file)
        output_file = os.path.join(f'{target_dir}/g16_input_files', f'{file.split(".xyz")[0]}.com')
        xyz_to_gaussian_input(xyz_file, output_file, method=method, basis_set=basis_set, extra_commands=extra_commands)


def process_reaction(idx, smiles_string, target_dir, reactive_complex_factor, bond_length_factor, disp_cut_off, freq_cut_off):
    ts_guess = None
    print(f'\nPath and preliminary TS search for reaction {idx}: {smiles_string}...')
    
    # If directory already exists, then replace it
    if f'reaction_{idx}' in os.listdir(target_dir):
        shutil.rmtree(f'{target_dir}/reaction_{idx}')
    
    change_workdir(f'{target_dir}/reaction_{idx}')
    
    try:
        ts_guess = obtain_ts_guess(smiles_string, reactive_complex_factor, bond_length_factor, disp_cut_off, freq_cut_off)
    except Exception as e:
        print(f'Error processing reaction {idx}: {str(e)}')
    
    if ts_guess is not None:
        print(f'TS for {smiles_string}: {ts_guess}')
        return f'reaction_{idx}'
    else:
        print(f'No TS found for {smiles_string}')
        return None


def get_guesses_from_smiles_list(smiles_strings, reactive_complex_factor, bond_length_factor, disp_cut_off, freq_cut_off):
    successful_reactions = []
    print(f'{len(smiles_strings)} reactions to process')

    num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
    pool = multiprocessing.Pool(processes=num_processes)

    process_reaction_with_args = partial(process_reaction, target_dir=target_dir,
                                     reactive_complex_factor=reactive_complex_factor,
                                     bond_length_factor=bond_length_factor,
                                     disp_cut_off=disp_cut_off,
                                     freq_cut_off=freq_cut_off)

    results = pool.starmap(process_reaction_with_args, enumerate(smiles_strings))

    pool.close()
    pool.join()

    print_statistics_guess_generation(successful_reactions, start_time)

    print("Successful reactions:", [r for r in results if r is not None])

    return successful_reactions


#    for idx, smiles_string in smiles_strings[:3]:
#        ts_guess = None
#        print(f'\nPath and preliminary TS search for reaction {idx}: {smiles_string}...')
#        # if directory already exists, then replace it
#        if f'reaction_{idx}' in os.listdir(target_dir):
#            shutil.rmtree(f'{target_dir}/reaction_{idx}')
#        change_workdir(f'{target_dir}/reaction_{idx}')
#        try:
#            ts_guess = obtain_ts_guess(smiles_string, reactive_complex_factor=reactive_complex_factor, \
#                                   bond_length_factor=bond_length_factor, disp_cut_off=disp_cut_off, \
#                                    freq_cut_off=freq_cut_off)
#        except:
#            continue
        
#        if ts_guess is not None:
#            print(f'TS for {smiles_string}: {ts_guess}')
#            successful_reactions.append(f'reaction_{idx}')
#        else:
#            print(f'No TS found for {smiles_string}')
    
#    print_statistics_guess_generation(successful_reactions, start_time)

#    print(successful_reactions)

#    return successful_reactions


def run_gaussian_jobs(folder):
    g16_work_dir = os.path.join(folder, 'g16_input_files')

    for filename in os.listdir(g16_work_dir):
        if filename.endswith(".com"):
            file_path = os.path.join(g16_work_dir, filename)
            
            if not os.path.exists(file_path):
                print(f"No Gaussian input file found in the folder: {g16_work_dir}")
                return 1
            
            # Run Gaussian 16 using nohup and redirect stderr to /dev/null
            log_file = os.path.splitext(file_path)[0] + ".log"
            out_file = os.path.splitext(file_path)[0] + ".out"
            with open(out_file, 'w') as out:
                    subprocess.run(
                        "g16 < {} > {}".format(os.path.join(g16_work_dir, filename), log_file),
                        shell=True,
                        stdout=out,
                        stderr=out,
                    )
                #command = ["g16", f"{file_path}", f"{log_file}"]
                #print(' '.join(command))
                #process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=f, stderr=subprocess.PIPE)
                #process.stdin.write(open(file_path).read().encode())
                #process.stdin.close()
                
                # Wait for the current job to complete
                #process.wait()


def confirm_opt_transition_states(folder, factor, disp_cut_off, freq_cut_off):
    g16_work_dir = os.path.join(folder, 'g16_input_files')

    for filename in os.listdir(g16_work_dir):
        if filename.endswith(".log"):
            # extract reaction name for folder location
            reaction_name = filename.split('_final')[0]
            afir_work_dir = os.path.join(folder, reaction_name)
            xyz_file_path = write_final_geometry_to_xyz(os.path.join(g16_work_dir, filename))
            print(xyz_file_path)

            #print(os.getcwd())
            # TODO: read charge from log file!
            # validate TS
            success = validate_ts_guess(xyz_file_path, afir_work_dir, factor=factor, disp_cut_off=disp_cut_off, 
                                        freq_cut_off=freq_cut_off, charge=0)
            if success:
                print(f'Good TS for {filename}')


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

        if final_geometry:
            with open(xyz_file_path, 'w') as xyz_file:
                num_atoms = len(final_geometry)
                xyz_file.write(f"{num_atoms}\n")
                xyz_file.write("Final geometry extracted from Gaussian log file\n")
                for atom_info in final_geometry:
                    xyz_file.write(f"{atom_info['symbol']} {atom_info['x']} {atom_info['y']} {atom_info['z']}\n")

    except FileNotFoundError:
        print(f"File not found: {log_file_path}")
    
    return xyz_file_path


if __name__ == "__main__":
    # settings
    reactive_complex_factor = 1.8
    bond_length_factor = 1.02 
    disp_cut_off = 0.7
    freq_cut_off = 150

    # preliminaries
    input_file = 'reactions_am.txt'
    target_dir = setup_dirs(f'benchmarking_{reactive_complex_factor}_{bond_length_factor}_{disp_cut_off}_{freq_cut_off}')
    smiles_strings = get_smiles_strings(input_file)
    start_time = time.time()

    # get all guesses
    successful_reactions = get_guesses_from_smiles_list(smiles_strings, reactive_complex_factor=reactive_complex_factor, \
                                   bond_length_factor=bond_length_factor, disp_cut_off=disp_cut_off, \
                                    freq_cut_off=freq_cut_off)

    generate_gaussian_inputs(target_dir, method='external="xtb_external.py"', basis_set='', 
                             extra_commands='opt=(ts, calcall, noeigen, nomicro)')

    run_gaussian_jobs(target_dir)

    confirm_opt_transition_states(target_dir, bond_length_factor, disp_cut_off, freq_cut_off)
