import logging

import numpy as np
import pandas as pd
import os
import shutil
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import time
from glob import glob

from tstools.ts_optimizer import TSOptimizer
from tstools.conformational_search import run_crest, run_g16_opt,  normal_termination_crest, run_g16_sp
from tstools.utils import remove_files_in_directory, copy_final_files, extract_g16_xtb_energy, read_active_atoms, extract_xtb_gibbs_free_energy, \
    write_stepwise_reactions_to_file, write_xyz_file_from_rdkit_conf, extract_crest_energy, extract_g16_energy, run_xtb_hessian, \
    extract_geom_from_xyz, write_xyz_file_from_geom, extract_geom_from_g16, extract_xtb_energy, get_charge_and_multiplicity

ps = Chem.SmilesParserParams()
ps.removeHs = False

logger = logging.getLogger("tstools")

class Reaction:

    def __init__(self, rxn_id, reaction_smiles, xtb_external_path, functional='UPBE1PBE',
                 basis_set='def2svp', mem='2GB', proc=2, max_cycles=30, dft_validation=False, gibbs=False, temperature=298.15):
        """
        Initialize a Reaction instance.

        Args:
            rxn_id (int): Reaction ID.
            reaction_smiles (str): SMILES representation of the reaction.
            xtb_external_path (str): Path to external xtb executable.
            logger (logging.Logger): logger-object
            mem (str, optional): Gaussian memory specification. Defaults to '2GB'.
            proc (int, optional): Number of processors. Defaults to 2.
            max_cycles (int, optional): Maximal number of cycles in TS geometry search. Defaults to 30.
            intermediate_check (bool): Whether or not to do an intermediate check.
            dft_validation (bool): Wheter or not obtain the reaction profile with DFT
        """
        self.rxn_id = rxn_id
        self.rxn_smiles = reaction_smiles
        self.reactant_smiles = reaction_smiles.split('>>')[0]
        self.product_smiles = reaction_smiles.split('>>')[-1]
        self.xtb_external_path = xtb_external_path
        self.functional = functional
        self.basis_set = basis_set
        self.mem = mem
        self.proc = proc
        self.max_cycles = max_cycles
        self.dft_validation = dft_validation
        self.gibbs = gibbs
        self.temp = temperature
        self.energies = {}
        self.ts_found = False

        self.reaction_dir = self.make_work_dir()

        self.conf_dir = self.make_sub_dir(sub_dir_name='conformers')
        self.rp_isolated_geometries_dir = self.make_sub_dir(sub_dir_name='rp_isolated_geometries')
        self.final_outputs = self.make_sub_dir(sub_dir_name='final_xtb')
        if dft_validation:
            self.dft_validation_ts_dir = self.make_sub_dir(sub_dir_name='dft_validation_ts')
            self.final_outputs_dft = self.make_sub_dir(sub_dir_name=os.path.join(self.final_outputs, 'final_dft'))

    def get_lowest_conformer_reacs_prods(self, num_conf, solvent_xtb, solvent_dft):

        """
        Conformational search of reactants and products

        Args:

            num_conf (int): Maximum number of conformers to optimize if dft_validation is set
            solvent_xtb (str): Solvent information for CREST calculation
            solvent_dft (str): Solvent information for DFT calculation

        Returns:

            None
        """

        for side, smiles in zip(['r', 'p'], [self.reactant_smiles, self.product_smiles]):
            for idx, smi in enumerate(smiles.split('.')):
                name = f"{side}{idx}"
                wcd = self.make_sub_dir(os.path.join(self.conf_dir, name)) #working conformer directory
                os.chdir(wcd)
                rdkit_mol = Chem.MolFromSmiles(smi, ps)
                charge, multiplicity = get_charge_and_multiplicity(smi)
                uhf = multiplicity - 1
                num_atoms = rdkit_mol.GetNumAtoms()
                logger.info(f"Conformational search of {name}")
                params = AllChem.ETKDGv3()
                params.randomSeed = 3
                AllChem.EmbedMultipleConfs(rdkit_mol, 1, params)
                write_xyz_file_from_rdkit_conf(rdkit_mol, os.path.join(wcd, name))
                run_crest(os.path.join(wcd, name), charge, uhf, None, self.proc, solvent_xtb, num_atoms)
                final_geom = 'crest_best.xyz' if num_atoms != 1 else f'{name}.xyz'
                if self.dft_validation:
                    run_g16_opt(os.path.join(wcd, name), num_conf, charge, uhf, None, self.mem, self.proc, self.functional, self.basis_set, solvent_dft)
                    shutil.copy(os.path.join(wcd, 'lowest_dft/dft_best.log'), f"{self.rp_isolated_geometries_dir}/{name}.log")
                    energy = extract_g16_energy(os.path.join(wcd, 'lowest_dft/dft_best.log'))
                else:
                    shutil.copy(os.path.join(wcd, final_geom), f"{self.rp_isolated_geometries_dir}/{name}.xyz")
                    if num_atoms !=1 :
                        energy = extract_crest_energy()
                    else:
                        energy = extract_xtb_energy(name)
                    if self.gibbs:
                        run_xtb_hessian(final_geom[:-4], charge, uhf, self.proc, solvent_xtb, self.temp)
                        shutil.move('crest_best_hess.out', f"{name}_hess.out")
                        energy = extract_xtb_gibbs_free_energy(f'{name}_hess')

                self.energies[name] = self.energies.get(name, 0.0) + energy

    def get_ts(self, reactive_complex_factor_values_inter=[2.5], reactive_complex_factor_values_intra=[1.1],
               freq_cut_off=150, solvent_xtb=None, solvent_dft=None, intermediate_check=False, add_broken_bonds=False):
        """
        Optimize an individual transition state. If a plausible intermediate is encountered
        along a reactive path twice, we assume a stepwise mechanism.

        Args:

            solvent_xtb (str): Solvent information for CREST calculation
            solvent_dft (str): Solvent information for DFT calculation
            reactive_complex_factor_values_inter (list): List of reactive complex factors for intermolecular reactions.
            reactive_complex_factor_values_intra (list): List of reactive complex factors for intramolecular reactions.
            freq_cut_off (int): Frequency cutoff.

        Returns:

            int, list or None: Reaction ID if a transition state is found, List of sub reactions if an intermediate is found, and None otherwise.
        """

        os.chdir(self.reaction_dir)
        ts_optimizer = TSOptimizer(self.rxn_id, self.rxn_smiles, self.xtb_external_path,
                                   solvent_xtb, solvent_dft, reactive_complex_factor_values_inter,
                                   reactive_complex_factor_values_intra, freq_cut_off, mem=self.mem, proc=self.proc,
                                   intermediate_check=intermediate_check, reaction_dir=self.reaction_dir, add_broken_bonds=add_broken_bonds)

        # First select the set of reactive_complex factor values to try
        start_time_process = time.time()
        stepwise_reaction_smiles = None

        try:
            if ts_optimizer.reaction_is_intramolecular():
                reactive_complex_factor_values = ts_optimizer.reactive_complex_factor_values_intra
            else:
                reactive_complex_factor_values = ts_optimizer.reactive_complex_factor_values_inter
        except Exception as e:
            logger.info(e)
            return None

        ts_found = False
        # Then search for TS by iterating through reactive complex factor values
        for reactive_complex_factor in reactive_complex_factor_values:
            if ts_found:
                break
            for i in range(2):
                logger.info(f'Attempt {i} with reactive complex factor = {reactive_complex_factor}')
                if ts_found:
                    break
                try:
                    ts_optimizer.set_ts_guess_list(reactive_complex_factor)
                    ts_optimizer.determine_ts()
                    ts_found = ts_optimizer.ts_found
                    remove_files_in_directory(os.getcwd())
                    if ts_found:
                        self.ts_found = True
                        end_time_process = time.time()
                        logger.info(
                            f'Final TS guess found for {ts_optimizer.rxn_id} for reactive complex factor {reactive_complex_factor} in {end_time_process - start_time_process} sec...')
                        
                except Exception as e:
                    logger.info(e)
                    continue

        end_time_process = time.time()

        if ts_optimizer.stepwise_reaction_smiles is not None:
            logger.info(
                f'Potential intermediate found for {ts_optimizer.rxn_id} for reactive complex factor {reactive_complex_factor} in {end_time_process - start_time_process} sec...')
            potentially_stepwise_reactions = [(f'{ts_optimizer.rxn_id}a', ts_optimizer.stepwise_reaction_smiles[0]),
                                              (f'{ts_optimizer.rxn_id}b', ts_optimizer.stepwise_reaction_smiles[1])]
            write_stepwise_reactions_to_file(potentially_stepwise_reactions, 'reaction.txt')
        if ts_found:
            for f in os.listdir(ts_optimizer.final_guess_dir):
                if '.log' in f:
                    energy_ts = extract_g16_xtb_energy(os.path.join(ts_optimizer.final_guess_dir, f))
                if self.gibbs:
                    cwd = os.getcwd()
                    os.chdir(ts_optimizer.g16_dir)
                    xyz_file = ts_optimizer.xyz_file
                    run_xtb_hessian(xyz_file[:-4], ts_optimizer.charge, ts_optimizer.multiplicity - 1, self.proc, solvent_xtb, self.temp)
                    energy_ts = extract_xtb_gibbs_free_energy(f'{xyz_file[:-4]}_hess')
                    os.chdir(cwd)
            self.energies['ts'] = self.energies.get('ts', 0.0) + energy_ts
            copy_final_files(self.reaction_dir, self.final_outputs) 
        else:
            logger.info(
                f'No TS guess found for {ts_optimizer.rxn_id}; process lasted for {end_time_process - start_time_process} sec...')

        if self.dft_validation and ts_found:
            ts_validated = False
            start_time_process = time.time()
            ts_optimizer_dft = TSOptimizer(self.rxn_id, self.rxn_smiles, None, xtb_solvent=solvent_xtb, dft_solvent=solvent_dft,
                                       guess_found=True, mem=self.mem, proc=self.proc, reaction_dir=self.dft_validation_ts_dir)

            guess_dir_path = os.path.join(self.reaction_dir, self.final_outputs)
            ts_optimizer_dft.path_dir = guess_dir_path
            ts_optimizer_dft.final_guess_dir = ts_optimizer_dft.reaction_dir

            try:
                for file in os.listdir(guess_dir_path):
                    if 'ts_guess' in file and file.endswith('.xyz'):
                        ts_guess_file = os.path.join(guess_dir_path, file)
                    elif file == 'reactants_geometry.xyz':
                        shutil.copy(os.path.join(guess_dir_path, file), ts_optimizer_dft.rp_geometries_dir)
                    elif file == 'products_geometry.xyz':
                        shutil.copy(os.path.join(guess_dir_path, file), ts_optimizer_dft.rp_geometries_dir)
            except Exception as e:
                print(e)

            if ts_guess_file is not None:
                ts_optimizer_dft.modify_ts_guess_list([ts_guess_file])
                try:
                    ts_optimizer_dft.determine_ts(xtb=False, method=self.functional,
                                                             basis_set=self.basis_set)
                    ts_validated = ts_optimizer_dft.ts_found
                    end_time_process = time.time()
                    if ts_validated:
                        logger.info(f"TS validated at {self.functional}/{self.basis_set} level of theory, "
                                    f"process lasted for {end_time_process - start_time_process} sec...'")
                    else:
                        logger.info("TS was not validated at DFT level of theory")
                        self.ts_found = False

                except Exception as e:
                    print(e)
                    pass

            if ts_validated:
                for file in os.listdir(ts_optimizer_dft.g16_dir):
                    if 'ts_guess' in file and not 'irc' in file and file.endswith(('xyz', 'log')):
                        ts_guess_file = os.path.join(ts_optimizer_dft.g16_dir, file)
                        shutil.copy(ts_guess_file, self.final_outputs_dft)
                        if file.endswith('log'):
                            self.energies['ts'] = extract_g16_energy(ts_guess_file)

        return None

    def get_lowest_ts(self, num_conf, solvent_xtb=None, solvent_dft=None, freq_cut_off=150):
        """
        Conformational search of TS

        Args:

            solvent_xtb (str): Solvent information for CREST calculation
            solvent_dft (str): Solvent information for DFT calculation
            freq_cut_off (int): Frequency cutoff.

        Returns:

            None
        """

        active_atoms = read_active_atoms(os.path.join(self.reaction_dir, 'final_ts_guess/active_atoms.inp'))
        if self.dft_validation:
            ts_guess_file = glob(f"{self.final_outputs_dft}/*.xyz")[0]
            final_guess_dir_path = os.path.join(self.reaction_dir, self.final_outputs_dft)
        else:
            ts_guess_file = glob(f"{os.path.join(self.reaction_dir, 'final_ts_guess')}/*.xyz")[0]
            final_guess_dir_path = os.path.join(self.reaction_dir, self.final_outputs)

        geom = extract_geom_from_xyz(ts_guess_file)
        wcd = self.make_sub_dir(os.path.join(self.conf_dir, 'ts'))  # working conformer directory
        os.chdir(wcd)
        logger.info("Conformational search of TS")
        write_xyz_file_from_geom(geom, os.path.join(wcd, 'ts_guess_conf'))
        rdkit_mol = Chem.MolFromSmiles(self.reactant_smiles, ps)
        charge = Chem.GetFormalCharge(rdkit_mol)
        uhf = Descriptors.NumRadicalElectrons(rdkit_mol)
        run_crest(os.path.join(wcd, 'ts_guess_conf'), charge, uhf, active_atoms, self.proc, solvent_xtb)

        if normal_termination_crest(os.path.join(wcd, 'ts_guess_conf.out')):

            if self.dft_validation:

                run_g16_sp(os.path.join(wcd, 'ts'), num_conf, charge, uhf, self.mem, self.proc, self.functional,
                           self.basis_set, solvent_dft)
                ts_conf_guess_geom = extract_geom_from_g16(os.path.join(wcd, 'lowest_dft/dft_best.log'))
                write_xyz_file_from_geom(ts_conf_guess_geom, os.path.join(wcd, 'ts_conf_dft'))
                ts_optimizer = TSOptimizer(self.rxn_id, self.rxn_smiles, None, None, dft_solvent=solvent_dft,
                                           freq_cut_off=freq_cut_off, guess_found=True, mem=self.mem, proc=self.proc,
                                           reaction_dir=wcd)

                guess_dir_path = os.path.join(self.reaction_dir, wcd)
                ts_optimizer.path_dir = guess_dir_path
                ts_optimizer.final_guess_dir = ts_optimizer.reaction_dir

                for file in os.listdir(os.path.join(self.reaction_dir, self.final_outputs)):
                    if file == 'reactants_geometry.xyz':
                        shutil.copy(os.path.join(os.path.join(self.reaction_dir, self.final_outputs), file), ts_optimizer.rp_geometries_dir)
                    elif file == 'products_geometry.xyz':
                        shutil.copy(os.path.join(os.path.join(self.reaction_dir, self.final_outputs), file), ts_optimizer.rp_geometries_dir)

                ts_guess_file = os.path.join(guess_dir_path, 'ts_conf_dft.xyz')

                ts_optimizer.modify_ts_guess_list([ts_guess_file])
                ts_optimizer.determine_ts(xtb=False, method=self.functional, basis_set=self.basis_set)
                ts_conf_found = ts_optimizer.ts_found

                if ts_conf_found:
                    logger.info('A conformer has been found for the TS')
                    shutil.copy(os.path.join("g16_dir", "ts_guess_0.xyz"), 
                                os.path.join(self.reaction_dir, "final_ts_guess/ts_guess_conf.xyz"))
                    shutil.copy(os.path.join("g16_dir", "ts_guess_0.log"),
                                os.path.join(self.reaction_dir, "final_ts_guess/ts_guess_conf.log"))
                    shutil.copy(os.path.join("g16_dir", "ts_guess_0.xyz"),
                                os.path.join(final_guess_dir_path, "ts_guess_conf.xyz"))
                    shutil.copy(os.path.join("g16_dir", "ts_guess_0.log"),
                                os.path.join(final_guess_dir_path, "ts_guess_conf.log"))
                    energy_ts = extract_g16_energy(
                        os.path.join(self.reaction_dir, "final_ts_guess/ts_guess_conf.log"))
                    if energy_ts < self.energies['ts']:
                        logger.info('TS conformer has lowest energy')
                        self.energies['ts'] = energy_ts
                    else:
                        logger.info('TS conformer does not have lowest energy')
                else:
                    logger.info("TS conformer guess is not a TS")

            else:
                ts_optimizer = TSOptimizer(self.rxn_id, self.rxn_smiles, self.xtb_external_path, solvent_xtb,
                                           None, freq_cut_off=freq_cut_off, guess_found=True, reaction_dir=wcd)

                guess_dir_path = os.path.join(self.reaction_dir, wcd)
                ts_optimizer.path_dir = guess_dir_path
                ts_optimizer.final_guess_dir = ts_optimizer.reaction_dir

                for file in os.listdir(os.path.join(self.reaction_dir, self.final_outputs)):
                    if file == 'reactants_geometry.xyz':
                        shutil.copy(os.path.join(final_guess_dir_path, file), ts_optimizer.rp_geometries_dir)
                    elif file == 'products_geometry.xyz':
                        shutil.copy(os.path.join(final_guess_dir_path, file), ts_optimizer.rp_geometries_dir)

                ts_guess_file = os.path.join(guess_dir_path, 'crest_best.xyz')

                ts_optimizer.modify_ts_guess_list([ts_guess_file])
                ts_optimizer.determine_ts()
                ts_conf_found = ts_optimizer.ts_found

                if ts_conf_found:
                    logger.info('A conformer has been found for the TS')
                    shutil.copy(os.path.join("g16_dir", "ts_guess_0.xyz"), 
                                os.path.join(self.reaction_dir, "final_ts_guess/ts_guess_conf.xyz"))
                    shutil.copy(os.path.join("g16_dir", "ts_guess_0.log"),
                                os.path.join(self.reaction_dir, "final_ts_guess/ts_guess_conf.log"))
                    shutil.copy(os.path.join("g16_dir", "ts_guess_0.xyz"), 
                                os.path.join(final_guess_dir_path, "ts_guess_conf.xyz"))
                    shutil.copy(os.path.join("g16_dir", "ts_guess_0.log"),
                                os.path.join(final_guess_dir_path, "ts_guess_conf.log"))
                    energy_ts = extract_g16_xtb_energy(os.path.join(self.reaction_dir, "final_ts_guess/ts_guess_conf.log"))
                    if self.gibbs:
                        os.chdir('g16_dir')
                        run_xtb_hessian('ts_guess_0', ts_optimizer.charge, ts_optimizer.multiplicity - 1, self.proc, solvent_xtb, self.temp)
                        energy_ts = extract_xtb_gibbs_free_energy('ts_guess_0_hess')
                        os.chdir(wcd)

                    if energy_ts < self.energies['ts']:
                        logger.info('TS conformer has lowest energy')
                        self.energies['ts'] = energy_ts
                    else:
                        logger.info('TS conformer does not have lowest energy')
                else:
                    logger.info("TS conformer guess is not a TS")

        else:
            logger.info('Error during conformational search of TS')


    def make_work_dir(self):
        """
        Create and return the main working directory.

        Returns:

            str: Path to the main working directory.
        """
        dir_name = f'reaction_{self.rxn_id}'
        if dir_name in os.listdir(os.getcwd()):
            shutil.rmtree(os.path.join(os.getcwd(), dir_name))
        os.makedirs(os.path.join(os.getcwd(), dir_name))

        return os.path.join(os.getcwd(), dir_name)

    def make_sub_dir(self, sub_dir_name):
        """
        Create and return a subdirectory within the reaction directory.

        Args:

            sub_dir_name (str): Name of the subdirectory.

        Returns:

            str: Path to the created subdirectory.
        """
        if sub_dir_name in os.listdir(self.reaction_dir):
            shutil.rmtree(os.path.join(self.reaction_dir, sub_dir_name))
        os.makedirs(os.path.join(self.reaction_dir, sub_dir_name))

        return os.path.join(self.reaction_dir, sub_dir_name)

    def make_plot(self):
        """
        Plot the reaction profile
        """

        logger.info("Plotting reaction profile")

        import matplotlib.pyplot as plt

        species = self.energies.keys()
        E_reacs = 0.0
        E_prods = 0.0
        for key in species:
            if 'r' in key:
                E_reacs += self.energies[key]
            if 'p' in key:
                E_prods += self.energies[key]

        if 'ts' in species:
            dE_ts = (self.energies['ts'] - E_reacs) * 627.509
        else:
            dE_ts = None

        base = 0.0
        dE_rxn = (E_prods - E_reacs) * 627.509

        if dE_ts:
            energies = [base, dE_ts, dE_rxn]
        else:
            energies = [base, dE_rxn]

        max_delta = 0.0
        min_energy = 0.0
        max_energy = 0.0

        fig, axes = plt.subplots(figsize=(4, 4))

        energies_duplicated = [energy for energy in energies for _ in range(2)]
        zi_s = np.array(range(len(energies_duplicated)))

        plt.plot(zi_s, energies_duplicated, linestyle="dashed", linewidth=1.0, color='gray')
        for i in energies:
            values = [i, i]
            idx = energies.index(i)
            xs = [2 * idx, 2 * idx + 1]
            plt.plot(xs, values, linewidth=2.5, color='black')

        for i, energy in enumerate(energies):
            axes.annotate(
               f"{np.round(energies_duplicated[i*2], 1)}",
               (zi_s[i*2] + 0.5, energies_duplicated[i*2] + 0.35),
               fontsize=10,
               ha="center",
            )

        max_delta_path = max(energies) - min(energies)
        if max_delta_path > max_delta:
            max_delta = max_delta_path
        if min_energy > min(energies):
            min_energy = min(energies)
        if max_energy < max(energies):
            max_energy = max(energies)

        plt.ylabel(f"$\Delta$E (kcal/mol)", fontsize=12)
        plt.xlabel("")
        plt.xticks([])
        plt.ylim(
            min_energy - 0.1 * max_delta,
            max_energy + 0.1 * max_delta,
        )
        plt.subplots_adjust(top=0.95, right=0.95)
        plt.tight_layout()
        plt.savefig(f"{self.reaction_dir}/reaction_profile.png", dpi=300)

    def compute_reaction_energy(self):

        species = self.energies.keys()
        E_reacs = 0.0
        E_prods = 0.0
        for key in species:
            if 'r' in key:
                E_reacs += self.energies[key]
            if 'p' in key:
                E_prods += self.energies[key]

        dE_rxn = (E_prods - E_reacs) * 627.509

        return dE_rxn

    def export_energy(self):
        """
                Export the energies of reactants, products and TSs to a .csv file.
        """

        df = pd.DataFrame.from_dict(self.energies, columns=['Energy'], orient='index')
        df.rename({'index': 'Species'}, inplace=True)
        df.to_csv(f'{self.reaction_dir}/energies.csv')


