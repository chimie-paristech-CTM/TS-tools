import time
import os
import shutil
import multiprocessing
import concurrent.futures

from reaction_profile_generator.ts_optimizer import TSOptimizer
from reaction_profile_generator.utils import remove_files_in_directory, copy_final_outputs


def get_reaction_list(filename):
    ''' a function that opens a file, reads in every line as a reaction smiles and returns them as a list. '''
    with open(filename, 'r') as f:
        lines = f.readlines()
        reaction_list = [line.rstrip().split() for line in lines]
    return reaction_list


def optimize_ts(ts_optimizer):
    # first select the set of reactive_complex factor values to try
    start_time_process = time.time()
    if ts_optimizer.reaction_is_intramolecular():
        reactive_complex_factor_values = ts_optimizer.reactive_complex_factor_values_intra
    else:
        reactive_complex_factor_values = ts_optimizer.reactive_complex_factor_values_inter
    #raise KeyError
    # then search for TS by iterating through reactive complex factor values
    for reactive_complex_factor in reactive_complex_factor_values:
        print(reactive_complex_factor)
        for _ in range(2):
            try:
                ts_optimizer.set_ts_guess_list(reactive_complex_factor)
                ts_found = ts_optimizer.determine_ts() 
                remove_files_in_directory(os.getcwd())
                if ts_found:
                    end_time_process = time.time()
                    print(f'Final TS guess found for {ts_optimizer.rxn_id} for reactive complex factor {reactive_complex_factor} in {end_time_process - start_time_process} sec...')
                    return ts_optimizer.rxn_id
            except Exception as e:
                print(e)
                continue
    
    end_time_process = time.time()
    print(f'No TS guess found for {ts_optimizer.rxn_id}; process lasted for {end_time_process - start_time_process} sec...')
    
    return None


def obtain_transition_states(target_dir, reaction_list, xtb_external_path, solvent, 
                             reactive_complex_factor_list_intermolecular, 
                             reactive_complex_factor_list_intramolecular, freq_cut_off):
    home_dir = os.getcwd()
    os.chdir(target_dir)
    ts_optimizer_list = []

    for rxn_idx, rxn_smiles in reaction_list:
        ts_optimizer_list.append(TSOptimizer(rxn_idx, rxn_smiles, xtb_external_path, 
                                             solvent, reactive_complex_factor_list_intermolecular,
                                             reactive_complex_factor_list_intramolecular, freq_cut_off))

    print(f'{len(ts_optimizer_list)} reactions to process...')

    num_processes = multiprocessing.cpu_count()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(num_processes/2)) as executor: 
        # Map the function to each object in parallel
        results = list(executor.map(optimize_ts, ts_optimizer_list))

    successful_reactions = [r for r in results if r is not None]

    os.chdir(home_dir)
    copy_final_outputs(target_dir)

    return successful_reactions


def setup_dir(target_dir):
    if target_dir in os.listdir():
        shutil.rmtree(target_dir)
    os.mkdir(target_dir)

    return target_dir


def print_statistics(successful_reactions, start_time):
    end_time = time.time()
    print(f'Successful reactions: {successful_reactions}')
    print(f'Number of successful reactions: {len(successful_reactions)}')
    print(f'Time taken: {end_time - start_time}')


def measure_time():
    obtain_transition_states(target_dir, reaction_list, xtb_external_path,
        solvent=solvent, reactive_complex_factor_list_intramolecular=reactive_complex_factor_list_intramolecular, 
        reactive_complex_factor_list_intermolecular = reactive_complex_factor_list_intermolecular, 
        freq_cut_off=freq_cut_off)
    

if __name__ == "__main__":
    # settings
    reactive_complex_factor_list_intramolecular = [1.2, 1.3, 1.4, 1.6, 1.8]
    reactive_complex_factor_list_intermolecular = [2.0]#[2.5, 1.8, 2.8, 2.6, 3.0, 2.3]
    freq_cut_off = 150
    solvent = None #'water' #None
    xtb_external_path = '"/home/thijs/Jensen_xtb_gaussian/profiles_test/extra/xtb_external.py"'

    # preliminaries
    input_file = 'reactions_am.txt'
    target_dir = setup_dir(f'benchmarking_{freq_cut_off}')
    reaction_list = get_reaction_list(input_file)[:1]
    #reaction_list = [['R1','[H:1][C:2]([H:3])=[C:4]([H:5])[H:6].[H:7][C:8]([H:9])=[C:10]([H:11])[H:12].[H:13][C:14]([H:15])=[C:16]([H:17])[H:18]>>[C:2]1([H:1])([H:3])[C:4]([H:5])([H:6])[C:8]([H:7])([H:9])[C:10]([H:11])([H:12])[C:14]([H:13])([H:15])[C:16]([H:17])([H:18])1'],
    #                 ['R2', '[H:1][C:2]([H:3])=[O:4].[H:5][O:6][H:7].[H:8][C:9]([H:10])=[C:11]([H:12])[O:13][H:14]>>[H:8][C:9]([H:10])([C:2]([H:1])([H:3])[O:4][H:5])[C:11]([H:12])=[O:13].[H:14][O:6][H:7]']]
    #    ['R3','[H:8][C:1]#[N:2].[H:7][C:3]#[N:4].[C-:5]#[N:6]>>[C-:3]#[N:4].[C:1](=[N:2][H:7])([C:5]#[N:6])[H:8]']]
    start_time = time.time()

    successful_reactions = obtain_transition_states(target_dir, reaction_list, xtb_external_path,
        solvent=solvent, reactive_complex_factor_list_intramolecular=reactive_complex_factor_list_intramolecular, 
        reactive_complex_factor_list_intermolecular = reactive_complex_factor_list_intermolecular, 
        freq_cut_off=freq_cut_off)

    print_statistics(successful_reactions, start_time)
