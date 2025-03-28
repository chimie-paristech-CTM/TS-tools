import time
import os
import argparse
from tstools.reaction import Reaction
from tstools.utils import setup_dir, get_reaction_list, create_logger


def get_args():
    """
    Parse command-line arguments.

    Returns:

    - argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--reactive-complex-factors-intra', nargs='+', type=float,
                        default=[0, 1.2, 1.3, 1.8])
    parser.add_argument('--num-conformers', action='store', type=int,
                        default=10)
    parser.add_argument('--nproc', action='store', type=int,
                        default=24)
    parser.add_argument('--mem', action='store', type=str,
                        default='24GB')
    parser.add_argument('--reactive-complex-factors-inter', nargs='+', type=float,
                        default=[2.5, 1.8, 2.8, 1.2])
    parser.add_argument('--freq-cut-off', action='store', type=int, default=50)
    parser.add_argument('--solvent-xtb', action='store', type=str, default=None)
    parser.add_argument('--solvent-dft', action='store', type=str, default=None)
    parser.add_argument('--xtb-external-path', action='store', type=str,
                        default="xtb_external_script/xtb_external.py")
    parser.add_argument('--input-file', action='store', type=str)
    parser.add_argument('--rxn-smiles', action='store', type=str)
    parser.add_argument('--target-dir', action='store', type=str, default='work_dir')
    parser.add_argument('--dft-validation', dest='dft_validation', action='store_true', default=False)
    parser.add_argument('--functional', action='store', type=str, default='UPBE1PBE')
    parser.add_argument('--basis-set', action='store', type=str, default='def2svp')

    return parser.parse_args()


def run_individual_reaction(idx, rxn_smiles, xtb_external_path, logger, solvent_xtb, solvent_dft, dft_validation,
                            reactive_complex_factors_intra, reactive_complex_factors_inter, num_conf, nproc, mem,
                            functional, basis_set):

    rxn = Reaction(idx, rxn_smiles, xtb_external_path, logger, functional=functional, basis_set=basis_set, mem=mem,
                   proc=nproc, dft_validation=dft_validation)

    t0 = time.time()
    rxn.get_lowest_conformer_reacs_prods(num_conf, solvent_xtb, solvent_dft)
    t_conf = time.time()

    logger.info(f"Conformational search for reaction {idx} lasts {t_conf - t0} sec")
    
    rxn.get_ts(reactive_complex_factor_values_intra=reactive_complex_factors_intra,
                   reactive_complex_factor_values_inter=reactive_complex_factors_inter,
                   solvent_xtb=solvent_xtb,
                   solvent_dft=solvent_dft)
    
    t_conf_ts_0 = time.time()
    if rxn.ts_found:
        rxn.get_lowest_ts(num_conf, solvent_xtb, solvent_dft)
    t_conf_ts_1 = time.time()

    logger.info(f"Conformational search for TS of reaction {idx} lasts {t_conf_ts_1 - t_conf_ts_0} sec")

    rxn.make_plot()
    rxn.export_energy()
    t1 = time.time()

    logger.info(f"Full reaction profile for reaction {idx} lasts {t1 - t0} sec")



if __name__ == "__main__":
    # preliminaries
    args = get_args()
    setup_dir(args.target_dir)
    xtb_external_path = f'{os.path.join(os.getcwd(), args.xtb_external_path)}'
    start_time = time.time()

    logger = create_logger()

    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    if not args.rxn_smiles and not args.input_file:
        logger.info('Nothing to do')
    else:
        if args.rxn_smiles:
            reaction_list = [args.rxn_smiles]
        else:
            reaction_list = get_reaction_list(args.input_file)

        home_dir = os.getcwd()
        os.chdir(args.target_dir)
        pwd = os.getcwd()

        for idx, rxn_smiles in reaction_list:
            run_individual_reaction(idx, rxn_smiles, xtb_external_path, logger, args.solvent_xtb, args.solvent_dft, args.dft_validation,
                                    args.reactive_complex_factors_intra, args.reactive_complex_factors_inter, args.num_conformers,
                                    args.nproc, args.mem, args.functional, args.basis_set)
            os.chdir(pwd)

    end_time = time.time()
    logger.info(f'Total time: {end_time - start_time} seconds')
