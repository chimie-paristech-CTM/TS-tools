import time
import os
import multiprocessing
import concurrent.futures
import argparse

from tstools.ts_optimizer import TSOptimizer
from tstools.utils import remove_files_in_directory, copy_final_outputs, \
    setup_dir, get_reaction_list, print_statistics, DetectedIntermediateException, \
    write_stepwise_reactions_to_file

def get_args():
    """
    Parse command-line arguments.

    Returns:
    - argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--reactive-complex_factors-intra', nargs='+', type=float,
                        default=[0, 1.2, 1.3, 1.8])
    parser.add_argument('--reactive-complex-factors-inter', nargs='+', type=float, 
                        default=[2.5, 1.8, 1.5, 1.3])
    parser.add_argument('--freq-cut-off', action='store', type=int, default=150)
    parser.add_argument('--solvent', action='store', type=str, default=None)
    parser.add_argument('--xtb-solvent', action='store', type=str, default=None)
    parser.add_argument('--dft-solvent', action='store', type=str, default=None)
    parser.add_argument('--xtb-external-path', action='store', type=str, 
                        default="xtb_external_script/xtb_external.py")
    parser.add_argument('--input-file', action='store', type=str, default='reactions_am.txt')
    parser.add_argument('--target-dir', action='store', type=str, default='work_dir_dft')
    parser.add_argument('--mem', action='store', type=str, default='16GB')
    parser.add_argument('--proc', action='store', type=int, default=8)
    parser.add_argument('--functional', action='store', type=str, default='UB3LYP')
    parser.add_argument('--basis-set', action='store', type=str, default='6-31G**')
    parser.add_argument('--max-cycles', action='store', type=int, default=30)
    parser.add_argument('--intermediate-check', dest='intermediate_check', action='store_true')
    parser.add_argument('--no-stepwise-mechanism-search', dest='stepwise_mechanism_search', action='store_false')

    args = parser.parse_args()

    # If general solvent specified, and not separately for the xTB and DFT calculations, set the latter
    if args.solvent is not None and (args.xtb_solvent is None and args.dft_solvent is None):
        args.xtb_solvent = args.solvent
        args.dft_solvent = args.solvent

    return args


def optimize_individual_ts(ts_optimizer):
    """
    Optimize an individual transition state.

    Parameters:
    - ts_optimizer: Instance of TSOptimizer.

    Returns:
    - int or None: Reaction ID if a transition state is found, None otherwise.
    """
    # First select the set of reactive_complex factor values to try
    start_time_process = time.time()
    detected_intermediate_timer = 0

    try:    
        if ts_optimizer.reaction_is_intramolecular():
            reactive_complex_factor_values = ts_optimizer.reactive_complex_factor_values_intra
        else:
            reactive_complex_factor_values = ts_optimizer.reactive_complex_factor_values_inter
    except Exception as e:
        print(e)
        return None

    # Then search for TS by iterating through reactive complex factor values
    for reactive_complex_factor in reactive_complex_factor_values:
        for _ in range(2):
            try:
                ts_optimizer.set_ts_guess_list(reactive_complex_factor)
                ts_optimizer.determine_ts(xtb=False, method=args.functional, basis_set=args.basis_set) 
                remove_files_in_directory(os.getcwd())
                if ts_optimizer.ts_found:
                    end_time_process = time.time()
                    print(f'Final TS guess found for {ts_optimizer.rxn_id} for reactive complex factor {reactive_complex_factor} in {end_time_process - start_time_process} sec...')                    
                    break
            except Exception as e:
                print(e)
                continue
        if ts_optimizer.ts_found:
            break

    end_time_process = time.time()

    if ts_optimizer.stepwise_reaction_smiles is not None:
        print(f'Potential intermediate found for {ts_optimizer.rxn_id} for reactive complex factor {reactive_complex_factor} in {end_time_process - start_time_process} sec...')

    if not ts_optimizer.ts_found:
        print(f'No TS guess found for {ts_optimizer.rxn_id}; process lasted for {end_time_process - start_time_process} sec...')

    return ts_optimizer


def obtain_transition_states(target_dir, reaction_list, xtb_external_path, xtb_solvent, dft_solvent,
                             reactive_complex_factor_list_intermolecular,
                             reactive_complex_factor_list_intramolecular, freq_cut_off, intermediate_check,
                             mem='16GB', proc=8, max_cycles=30):
    """
    Obtain transition states for a list of reactions.

    Parameters:
    - target_dir (str): Target directory.
    - reaction_list (list): List of reactions.
    - xtb_external_path (str): Path to the XTB external script.
    - xtb_solvent (str): Solvent information for xTB calculations.
    - dft_solvent (str): Solvent information for DFT calculations.
    - reactive_complex_factor_list_intermolecular (list): List of reactive complex factors for intermolecular reactions.
    - reactive_complex_factor_list_intramolecular (list): List of reactive complex factors for intramolecular reactions.
    - freq_cut_off (int): Frequency cutoff.
    - intermediate_check (bool): Whether or not to do an intermediate check.
    - mem (str, optional): Amount of memory to allocate for the calculations (default is '16GB').
    - proc (int, optional): Number of processor cores to use for the calculations (default is 8).
    - max_cycles (int, optional): Maximal number of cycles in TS geometry search (default is 30).

    Returns:
    - list: List of successful reactions.
    """
    home_dir = os.getcwd()
    os.chdir(target_dir)
    ts_optimizer_list = []

    for rxn_idx, rxn_smiles in reaction_list:
        ts_optimizer_list.append(TSOptimizer(rxn_idx, rxn_smiles, xtb_external_path,
                                             xtb_solvent, dft_solvent, reactive_complex_factor_list_intermolecular,
                                             reactive_complex_factor_list_intramolecular, freq_cut_off,
                                             mem=mem, proc=proc, max_cycles=max_cycles, intermediate_check=intermediate_check))

    print(f'{len(ts_optimizer_list)} reactions to process...')

    num_processes = multiprocessing.cpu_count()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Map the function to each object in parallel
        results = list(executor.map(optimize_individual_ts, ts_optimizer_list))

    successful_reactions, potentially_stepwise_reactions = [], []

    for result in results:
        if result.stepwise_reaction_smiles is not None:
                potentially_stepwise_reactions += [f'{result.rxn_id}a', result.stepwise_reaction_smiles[0]], [f'{result.rxn_id}b', result.stepwise_reaction_smiles[1]]
        if result.ts_found:
            successful_reactions.append(result.rxn_id)

    os.chdir(home_dir)
    copy_final_outputs(target_dir, f'final_{target_dir}')

    return successful_reactions, potentially_stepwise_reactions
    

def run_all_reaction_smiles_in_file(args, input_file):
    """
    Process a file containing reaction SMILES strings to calculate transition states.

    Parameters:
    - args: Namespace
        Command-line arguments or an object containing parameters such as target directory, solvent, 
        XTB external path, reactive complex factors, frequency cutoff, and intermediate check options.
    - input_file: str
        Path to the input file containing reaction SMILES strings, each on a separate line.

    Returns:
    - successful_reactions: list
        A list of successfully optimized reactions with transition states identified.
    - stepwise_reactions: list
        A list of reactions identified as stepwise, based on intermediate checks.
    """
    reaction_list = get_reaction_list(input_file)
    xtb_external_path = f'{os.path.join(os.getcwd(), args.xtb_external_path)}'

    successful_reactions, stepwise_reactions = obtain_transition_states(args.target_dir, reaction_list, 
        xtb_external_path, xtb_solvent=args.xtb_solvent, dft_solvent=args.dft_solvent,
        reactive_complex_factor_list_intramolecular=args.reactive_complex_factors_intra, 
        reactive_complex_factor_list_intermolecular=args.reactive_complex_factors_inter, 
        freq_cut_off=args.freq_cut_off, intermediate_check=args.intermediate_check, 
        mem=args.mem, proc=args.proc, max_cycles=args.max_cycles)

    return successful_reactions, stepwise_reactions 


if __name__ == "__main__":
    # preliminaries
    args = get_args()
    setup_dir(args.target_dir)
    start_time = time.time()

    # run all reactions in parallel
    successful_reactions, potentially_stepwise_reactions = run_all_reaction_smiles_in_file(args, args.input_file)
    print_statistics(successful_reactions, potentially_stepwise_reactions, start_time)

    if len(potentially_stepwise_reactions) > 0:
        stepwise_reaction_file = write_stepwise_reactions_to_file(potentially_stepwise_reactions, args.input_file)
