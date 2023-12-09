import time
import os
import shutil
import subprocess
import multiprocessing

from reaction_profile_generator.path_generator import PathGenerator
from reaction_profile_generator.utils import work_in, xyz_to_gaussian_input
from reaction_profile_generator.irc_search import generate_gaussian_irc_input, extract_transition_state_geometry, extract_irc_geometries, compare_molecules_irc


def get_reaction_list(filename):
    ''' a function that opens a file, reads in every line as a reaction smiles and returns them as a list. '''
    with open(filename, 'r') as f:
        lines = f.readlines()
        reaction_list = [line.rstrip().split() for line in lines]
    return reaction_list

def obtain_ts_guess(reaction_smiles, reactive_complex_factor=1.8, freq_cut_off=150, solvent=None):
    ''' a function that splits up a reaction smiles in reactant and product, and then calls the function get_path with these as parameters. '''
    reactant_smiles, product_smiles = reaction_smiles.split('>>')
    reaction = PathGenerator(reactant_smiles, product_smiles, solvent, reactive_complex_factor, freq_cut_off)
    if len(reaction.formed_bonds) < len(reaction.broken_bonds):
        reaction = PathGenerator(product_smiles, reactant_smiles, solvent, reactive_complex_factor, freq_cut_off)
    success = reaction.get_ts_guesses_from_path()

    return success


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
    pool = multiprocessing.Pool(processes=int(num_processes/2))

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


def optimize_ts_guesses(folder, reaction_id):
    g16_work_dir = os.path.join(folder, 'g16_work_dir')
    if reaction_id not in os.listdir(g16_work_dir):
        return False
    else:
        ts_input_files_list = [filename for filename in os.listdir(os.path.join(g16_work_dir, reaction_id)) if filename.endswith(".com")]
        sorted_ts_input_files_list = sorted(ts_input_files_list)

        for filename in sorted_ts_input_files_list:
            file_path = os.path.join(os.path.join(g16_work_dir, reaction_id), filename)
            
            if not os.path.exists(file_path):
                print(f"No Gaussian input file found in the folder: {g16_work_dir}")
                break
            
            log_file = run_g16_ts_optimization(file_path)

            success = confirm_opt_transition_state(log_file, folder, reaction_id)

            print(log_file, success)

            if success:
                return True
    
    return False


def optimize_all_ts_guesses(folder, reaction_list):
    successful_ts_list = []
    for reaction in reaction_list:
        success = optimize_ts_guesses(folder, f'reaction_{reaction[0]}')
        if success:
            successful_ts_list.append(f'reaction_{reaction[0]}')

    return successful_ts_list


def confirm_opt_transition_state(log_file, target_dir, directory):
    try:
        extract_transition_state_geometry(log_file, f'{log_file[:-4]}.xyz')
        irc_input_file_f, irc_input_file_r = generate_gaussian_irc_input(f'{log_file[:-4]}.xyz', output_prefix=f'{log_file.split("/")[-1][:-4]}_irc',
            method='external="/home/thijs/Jensen_xtb_gaussian/profiles_test/extra/xtb_external.py"')
        print(irc_input_file_f, irc_input_file_r)
        run_irc(irc_input_file_f)
        run_irc(irc_input_file_r)
        extract_irc_geometries(f'{irc_input_file_f[:-4]}.log', f'{irc_input_file_r[:-4]}.log')
        reaction_correct = compare_molecules_irc(
            f'{irc_input_file_f[:-4]}.xyz',
            f'{irc_input_file_r[:-4]}.xyz',
            f'{target_dir}/rp_geometries/{directory}/reactants_geometry.xyz', 
            f'{target_dir}/rp_geometries/{directory}/products_geometry.xyz'
        )
        if reaction_correct:
            return True
        else:
            return False
    except:
        return False


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


def update_reaction_list(reaction_list, successful_reactions_final):
    reactions_still_to_do = []
    successful_ids = set([reaction.split('_')[-1] for reaction in successful_reactions_final])

    for reaction in reaction_list:
        if reaction[0] not in successful_ids:
            reactions_still_to_do.append(reaction)

    return reactions_still_to_do


def remove_g16_work_folders(folder, reactions_finished):
    g16_work_dir = os.path.join(folder, 'g16_work_dir')
    for dir_name in os.listdir(g16_work_dir):
        print(dir_name, reactions_finished)
        if dir_name not in reactions_finished:
            shutil.rmtree(os.path.join(g16_work_dir, dir_name))
        else:
            continue


if __name__ == "__main__":
    # settings
    reactive_complex_factor_list = [2.4] #[2.4, 1.8, 3.0, 2.6, 2.1]
    freq_cut_off = 150

    # preliminaries
    input_file = 'reactions_am.txt'
    target_dir = setup_dirs(f'benchmarking_{freq_cut_off}')
    reaction_list = get_reaction_list(input_file)[:4]
    start_time = time.time()


    reactions_finished = []
    # get all guesses
    for reactive_complex_factor in reactive_complex_factor_list:

        successful_reactions = get_guesses_from_smiles_list(reaction_list, reactive_complex_factor=reactive_complex_factor, freq_cut_off=freq_cut_off, 
                                                            start_time=start_time)

        generate_gaussian_inputs(target_dir, method='external="/home/thijs/Jensen_xtb_gaussian/profiles_test/extra/xtb_external.py"', basis_set='', 
                             extra_commands='opt=(ts, calcall, noeigen, nomicro)')

        successful_reactions_final = optimize_all_ts_guesses(target_dir, reaction_list)   
        reactions_finished += successful_reactions_final

        reaction_list = update_reaction_list(reaction_list, successful_reactions_final)

        print(reactions_finished)

        remove_g16_work_folders(target_dir, reactions_finished) # remove folders for reactions that have failed

        if len(reaction_list) == 0:
            break

    print_statistics(reactions_finished, start_time)
