import time
import os
import shutil
import subprocess
import multiprocessing

from reaction_profile_generator.get_path import determine_ts_guesses_from_path
from reaction_profile_generator.utils import work_in, xyz_to_gaussian_input
from reaction_profile_generator.irc_search import generate_gaussian_irc_input, extract_transition_state_geometry, extract_irc_geometries, compare_molecules_irc

# TODO: sort out issue with workdir!
workdir = ['test']


def change_workdir(new_name):
    workdir[0] = new_name


def get_reaction_list(filename):
    ''' a function that opens a file, reads in every line as a reaction smiles and returns them as a list. '''
    with open(filename, 'r') as f:
        lines = f.readlines()
        reaction_list = [line.rstrip().split() for line in lines]
    return reaction_list


def get_reaction_list_alt():
    return [#['R73','[C:1](=[C:2]([C:3](=[C:4](\[H:14])[H:15])\[H:16])/[H:7])(\[H:8])[H:9].[C:5](=[C:6](/[H:12])[H:13])(\[H:10])[H:11]>>[C:1]1([H:8])([H:9])[C:2]([H:7])=[C:3]([H:16])[C:4]([H:14])([H:15])[C:6]([H:12])([H:13])[C:5]1([H:10])[H:11]'],
        ['R74','[C@:1]1([C:5]([H:11])([H:12])[H:13])([H:7])[C:2]([H:8])=[C:3]([H:9])[C@@:4]1([C:6]([H:14])([H:15])[H:16])[H:10]>>[C:1](=[C:2]([C:3](=[C:4](/[C:6]([H:14])([H:15])[H:16])[H:10])/[H:9])\[H:8])(\[C:5]([H:11])([H:12])[H:13])[H:7]']]


@work_in(workdir)
def obtain_ts_guess(reaction_smiles, reactive_complex_factor=1.8, freq_cut_off=150):
    ''' a function that splits up a reaction smiles in reactant and product, and then calls the function get_path with these as parameters. '''
    reactant_smiles, product_smiles = reaction_smiles.split('>>')
    ts_guesses_found = determine_ts_guesses_from_path(reactant_smiles, product_smiles, reactive_complex_factor=reactive_complex_factor, 
                                freq_cut_off=freq_cut_off) #, solvent='water')

    return ts_guesses_found

def setup_dirs(target_dir):
    if target_dir in os.listdir():
        shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    os.mkdir(f'{target_dir}/final_ts_guesses')
    os.mkdir(f'{target_dir}/g16_work_dir')
    os.mkdir(f'{target_dir}/rp_geometries')

    return target_dir


def print_statistics(successful_reactions, start_time):
    end_time = time.time()
    print(f'Successful reactions: {successful_reactions}')
    print(f'Number of successful reactions: {len(successful_reactions)}')
    print(f'Time taken: {end_time - start_time}')


def generate_gaussian_inputs(target_dir, method='external="../xtb_external.py"', basis_set='', 
                             extra_commands='opt=(ts, calcall, noeigen, nomicro)'):
    for dir in os.listdir(f'{target_dir}/final_ts_guesses'):
        if dir not in os.listdir(f'{target_dir}/g16_work_dir'):
            os.makedirs(f'{target_dir}/g16_work_dir/{dir}')
        for file in os.listdir(f'{target_dir}/final_ts_guesses/{dir}'):
            xyz_file = os.path.join(f'{target_dir}/final_ts_guesses/{dir}', file)
            output_file = os.path.join(f'{target_dir}/g16_work_dir/{dir}', f'{file.split(".xyz")[0]}.com')
            xyz_to_gaussian_input(xyz_file, output_file, method=method, basis_set=basis_set, extra_commands=extra_commands)


def process_reaction(args):
    ts_guesses_found = False
    reaction, target_dir, reactive_complex_factor, freq_cut_off = args
    idx, smiles_string = reaction
    print(f'\nPath and preliminary TS search for reaction {idx}: {smiles_string}...')
    
    # If directory already exists, then replace it
    if f'reaction_{idx}' in os.listdir(target_dir):
        shutil.rmtree(f'{target_dir}/reaction_{idx}')
    
    change_workdir(f'{target_dir}/reaction_{idx}')

    #try:
    ts_guesses_found = obtain_ts_guess(smiles_string, reactive_complex_factor, freq_cut_off)
    #except Exception as e:
    #    print(f'Error processing reaction {idx}: {str(e)}')
    
    if ts_guesses_found:
        print(f'TS guess found for {smiles_string}')
        return f'reaction_{idx}'
    else:
        print(f'No TS guess found for {smiles_string}')
        return None


def get_guesses_from_smiles_list(reaction_list, reactive_complex_factor, freq_cut_off, start_time):
    successful_reactions = []
    print(f'{len(reaction_list)} reactions to process')

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    args = list(zip(reaction_list, 
               [target_dir for _ in reaction_list],
               [reactive_complex_factor for _ in reaction_list],
               [freq_cut_off for _ in reaction_list]))
    
    results = pool.map(process_reaction, args)

    pool.close()
    pool.join()

    successful_reactions = [r for r in results if r is not None]

    print_statistics(successful_reactions, start_time)

    return successful_reactions


def run_gaussian_jobs(folder):
    g16_work_dir = os.path.join(folder, 'g16_work_dir')

    for dir in os.lisdir(g16_work_dir):
        for filename in os.listdir(os.path.join(g16_work_dir, dir)):
            if filename.endswith(".com"):
                file_path = os.path.join(os.path.join(g16_work_dir, dir), filename)
            
                if not os.path.exists(file_path):
                    print(f"No Gaussian input file found in the folder: {g16_work_dir}")
                    return 1
            
                # Run Gaussian 16 using nohup and redirect stderr to /dev/null
                log_file = os.path.splitext(file_path)[0] + ".log"
                out_file = os.path.splitext(file_path)[0] + ".out"
                with open(out_file, 'w') as out:
                    subprocess.run(
                        "g16 < {} >> {}".format(os.path.join(os.path.join(g16_work_dir, filename), dir), log_file),
                        shell=True,
                        stdout=out,
                        stderr=out,
                    )


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


def confirm_opt_transition_states(target_dir):
    successful_reactions_final = []

    for directory in f'{target_dir}/g16_work_dir':
        output_file_list = sorted([f'{target_dir}/g16_work_dir/{directory}/{file}' 
                            for file in os.listdir(f'{target_dir}/g16_work_dir/{directory}') 
                            if file.endswith('.log')])
        for file_path in output_file_list:
            extract_transition_state_geometry(file_path, f'{file_path[:-4]}.xyz')
            generate_gaussian_irc_input(f'{file_path[:-4]}.xyz', method='external="/home/thijs/Jensen_xtb_gaussian/profiles_test/extra/xtb_external.py"')
            # run_irc()
            extract_irc_geometries(f'{target_dir}/g16_work_dir/{directory}/irc_calc.log')
            reaction_correct = compare_molecules_irc(
                f'{target_dir}/g16_work_dir/{directory}/irc_calc_forward.xyz',
                f'{target_dir}/g16_work_dir/{directory}/irc_calc_reverse.xyz',
                f'{target_dir}/rp_geometries/{directory}/reactants_geometry.xyz', 
                f'{target_dir}/rp_geometries/{directory}/products_geometry.xyz'
            )
            
            if reaction_correct:
                successful_reactions_final.append(directory)
                break
            else:
                continue
    
    return successful_reactions_final

            
if __name__ == "__main__":
    # settings
    reactive_complex_factor = 2.0
    freq_cut_off = 200

    # preliminaries
    input_file = 'reactions_am.txt'
    target_dir = setup_dirs(f'benchmarking_{reactive_complex_factor}_{freq_cut_off}')
    reaction_list = get_reaction_list(input_file)[:3]
    start_time = time.time()

    # get all guesses
    successful_reactions = get_guesses_from_smiles_list(reaction_list, reactive_complex_factor=reactive_complex_factor, freq_cut_off=freq_cut_off, start_time=start_time)

    generate_gaussian_inputs(target_dir, method='external="/home/thijs/Jensen_xtb_gaussian/profiles_test/extra/xtb_external.py"', basis_set='', 
                             extra_commands='opt=(ts, calcall, noeigen, nomicro)')

    #run_gaussian_jobs(target_dir)

    #successful_reactions_final = confirm_opt_transition_states(target_dir)

    #print_statistics(successful_reactions_final)
