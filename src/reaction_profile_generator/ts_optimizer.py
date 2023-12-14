import numpy as np
import os
import shutil

from reaction_profile_generator.path_generator import PathGenerator
from reaction_profile_generator.confirm_ts_guess import validate_ts_guess
from reaction_profile_generator.utils import xyz_to_gaussian_input, run_g16_ts_optimization, run_irc, remove_files_in_directory
from reaction_profile_generator.irc_search import generate_gaussian_irc_input, extract_transition_state_geometry, extract_irc_geometries, compare_molecules_irc


class TSOptimizer:
    def __init__(self, rxn_id, reaction_smiles, xtb_external_path, solvent=None, 
                reactive_complex_factor_values_inter= [2.5], reactive_complex_factor_values_intra = [1.1], freq_cut_off=150):
        self.rxn_id = rxn_id
        self.reactant_smiles = reaction_smiles.split('>>')[0]
        self.product_smiles = reaction_smiles.split('>>')[-1]
        self.solvent = solvent
        self.reactive_complex_factor_values_inter = reactive_complex_factor_values_inter
        self.reactive_complex_factor_values_intra = reactive_complex_factor_values_intra
        self.freq_cut_off = freq_cut_off
        self.xtb_external_path = xtb_external_path

        self.reaction_dir = self.make_work_dir()

        self.path_dir = self.make_sub_dir(sub_dir_name='path_dir')
        self.ts_guesses_dir = self.make_sub_dir(sub_dir_name='preliminary_ts_guesses')
        self.g16_dir = self.make_sub_dir(sub_dir_name='g16_dir')
        self.rp_geometries_dir = self.make_sub_dir(sub_dir_name='rp_geometries')
        self.final_guess_dir = self.make_sub_dir(sub_dir_name='final_ts_guess')

        self.ts_guess_list = None

    def determine_ts(self):
        if self.ts_guess_list == None:
            return False
        
        remove_files_in_directory(self.g16_dir)

        for i, guess_file in enumerate(self.ts_guess_list):
            if self.solvent is not None:
                ts_search_inp_file = self.generate_g16_input_ts_opt(
                    i, guess_file, method=f'external={self.xtb_external_path}', basis_set='', 
                    extra_commands=f'opt=(ts, calcall, noeigen, nomicro) SCRF=(Solvent={self.solvent})')
            else:
                ts_search_inp_file = self.generate_g16_input_ts_opt(
                    i, guess_file, method=f'external={self.xtb_external_path}', basis_set='')
            os.chdir(self.reaction_dir)
            log_file = run_g16_ts_optimization(ts_search_inp_file)
            success = self.confirm_opt_transition_state(log_file)

            if success:
                xyz_file = f'{os.path.splitext(log_file)[0]}.xyz'
                self.save_final_ts_guess_files(xyz_file, log_file)
                return True
        
        return False

    def set_ts_guess_list(self, reactive_complex_factor):
        path = self.set_up_path_generator(reactive_complex_factor)
        ts_guess_list = self.obtain_ts_guesses_for_given_reactive_complex_factor(path)
        if ts_guess_list is not None:
            self.save_ts_guesses(ts_guess_list)

        self.ts_guess_list = ts_guess_list

    def set_up_path_generator(self, reactive_complex_factor, n_conf=100):
        path = PathGenerator(self.reactant_smiles, self.product_smiles, self.rxn_id, self.path_dir, self.rp_geometries_dir,
                             self.solvent, reactive_complex_factor, self.freq_cut_off, n_conf=n_conf)
        if len(path.formed_bonds) < len(path.broken_bonds):
            path = PathGenerator(self.product_smiles, self.reactant_smiles, self.rxn_id, self.path_dir, self.rp_geometries_dir,
                             self.solvent, reactive_complex_factor, self.freq_cut_off, n_conf=n_conf)

        return path

    def obtain_ts_guesses_for_given_reactive_complex_factor(self, path):
        for _ in range(5):
            energies, potentials, path_xyz_files = path.get_path()
            if energies is not None:
                true_energies = list(np.array(energies) - np.array(potentials))
                guesses_list = self.determine_and_filter_local_maxima(true_energies, path_xyz_files, path.charge)
                if len(guesses_list) > 0:
                    return guesses_list
        
        return None

    def determine_and_filter_local_maxima(self, true_energies, path_xyz_files, charge):
        # Find local maxima in path
        indices_local_maxima = find_local_max_indices(true_energies)

        # Validate the local maxima and store their energy values
        ts_guess_dict = {}
        idx_local_maxima_correct_mode = []
        for index in indices_local_maxima:
            ts_guess_file, _ = validate_ts_guess(path_xyz_files[index], self.reaction_dir, self.freq_cut_off, charge)
            if ts_guess_file is not None:
                idx_local_maxima_correct_mode.append(index)
                ts_guess_dict[ts_guess_file] = true_energies[index]

        # Sort guesses based on energy
        sorted_guess_dict = sorted(ts_guess_dict.items(), key=lambda x: x[1], reverse=True)
        ranked_guess_files = [item[0] for item in sorted_guess_dict]

        return ranked_guess_files

    def make_work_dir(self):
        dir_name = f'reaction_{self.rxn_id}'
        if dir_name in os.listdir(os.getcwd()):
           shutil.rmtree(os.path.join(os.getcwd(), dir_name)) 
        os.makedirs(os.path.join(os.getcwd(), dir_name))

        return os.path.join(os.getcwd(), dir_name)
    
    def make_sub_dir(self, sub_dir_name):
        if sub_dir_name in os.listdir(self.reaction_dir):
           shutil.rmtree(os.path.join(self.reaction_dir, sub_dir_name)) 
        os.makedirs(os.path.join(self.reaction_dir, sub_dir_name))

        return os.path.join(self.reaction_dir, sub_dir_name)

    def save_ts_guesses(self, ts_guesses_list):
        for i, ts_guess_file in enumerate(ts_guesses_list):
            shutil.copy(ts_guess_file, self.ts_guesses_dir)
            os.rename(
                os.path.join(self.ts_guesses_dir, os.path.basename(ts_guess_file)),
                os.path.join(self.ts_guesses_dir, f'ts_guess_{i}.xyz')
            )
    
    def generate_g16_input_ts_opt(self, idx, file_name, method, basis_set='', 
                             extra_commands='opt=(ts, calcall, noeigen, nomicro)'):
        ts_search_inp_file = os.path.join(self.g16_dir, f'ts_guess_{idx}.com')
        xyz_to_gaussian_input(os.path.join(self.path_dir, file_name), ts_search_inp_file, method=method, basis_set=basis_set, extra_commands=extra_commands)

        return ts_search_inp_file

    def confirm_opt_transition_state(self, log_file):
        try:
            extract_transition_state_geometry(log_file, f'{os.path.splitext(log_file)[0]}.xyz')
            irc_input_file_f, irc_input_file_r = generate_gaussian_irc_input(f'{os.path.splitext(log_file)[0]}.xyz', 
                output_prefix=f'{os.path.splitext(log_file)[0]}_irc', method=f'external={self.xtb_external_path}', solvent=self.solvent)
            run_irc(irc_input_file_f)
            run_irc(irc_input_file_r)
            extract_irc_geometries(f'{os.path.splitext(irc_input_file_f)[0]}.log', f'{os.path.splitext(irc_input_file_r)[0]}.log')
            reaction_correct = compare_molecules_irc(
                f'{os.path.splitext(irc_input_file_f)[0]}.xyz',
                f'{os.path.splitext(irc_input_file_r)[0]}.xyz',
                os.path.join(self.rp_geometries_dir, 'reactants_geometry.xyz'),
                os.path.join(self.rp_geometries_dir, 'products_geometry.xyz'),
                self.solvent
            )
            if reaction_correct:
                return True
            else:
                return False
        except:
            return False
        
    def reaction_is_intramolecular(self):
        path = self.set_up_path_generator(reactive_complex_factor=1.2, n_conf=1)
        if len(path.reactant_smiles.split('.')) == 1:
            return True
        else:
            return False
 
    def save_final_ts_guess_files(self, xyz_file, log_file):
        shutil.copy(xyz_file, self.final_guess_dir)
        shutil.copy(log_file, self.final_guess_dir)


def find_local_max_indices(numbers):
    local_max_indices = []
    for i in range(len(numbers) - 2, 0, -1):
        if numbers[i] > numbers[i - 1] and numbers[i] > numbers[i + 1]:
            local_max_indices.append(i)
    return local_max_indices


if __name__ == '__main__':
    pass
