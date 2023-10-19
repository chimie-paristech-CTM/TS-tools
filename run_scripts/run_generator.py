from reaction_profile_generator.get_path import get_path
from reaction_profile_generator.relax_path import generate_ts_guess
from rdkit import Chem
from rdkit.Chem import AllChem
from reaction_profile_generator.utils import work_in, xyz_to_gaussian_input
import time
import os
import shutil

# TODO: sort out issue with workdir!
workdir = ['test']


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
def run_path_search(reaction_smiles):
    ''' a function that splits up a reaction smiles in reactant and product, and then calls the function get_path with these as parameters. '''
    reactant_smiles, product_smiles = reaction_smiles.split('>>')
    path_xyz_files = get_path(reactant_smiles, product_smiles) #, solvent='water')

    return path_xyz_files


@work_in(workdir)
def run_path_relaxation(xyz_files, prod_distances, reac_distances):
    ''' a function that takes in the list of xyz-files of a path and then relaxes the path by calling the relax_path method. '''
    path_xyz_files = generate_ts_guess(xyz_files, prod_distances, reac_distances) #, solvent='water')

    return path_xyz_files


if __name__ == "__main__":
    input_file = 'reactions_am.txt'
    target_dir = 'benchmarkingxxxxxxxxxxx'
    if target_dir in os.listdir():
        shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    os.mkdir(f'{target_dir}/final_ts_guesses')
    os.mkdir(f'{target_dir}/g16_input_files')

    smiles_strings = get_smiles_strings(input_file)
    #smiles_strings = get_smiles_strings_alt()
    start_time = time.time()
    successful_reactions = []

    for idx, smiles_string in smiles_strings[20:30]: #[20:23]: #[16:18]):
        ts_guess = None
        for i in range(10):
            # if directory already exists, then replace it
            if f'reaction_{idx}' in os.listdir(target_dir):
                shutil.rmtree(f'{target_dir}/reaction_{idx}')
            change_workdir(f'{target_dir}/reaction_{idx}')
            try:
                path_xyzs, prod_distances, reac_distances = run_path_search(smiles_string)
                ts_guess = run_path_relaxation(path_xyzs, prod_distances, reac_distances)
                if ts_guess is not None:
                    break
                else: 
                    continue
            except:
                continue
        
        if ts_guess is not None:
            print(f'TS for {smiles_string}: {ts_guess}')
            successful_reactions.append(f'reaction_{idx}')
        else:
            print(f'No TS found for {smiles_string}')

    end_time = time.time()
    print(f'Successful reactions: {successful_reactions}')
    print(f'Number of successful reactions: {len(successful_reactions)}')
    print(f'Time taken: {end_time - start_time}')

    for file in os.listdir(f'{target_dir}/final_ts_guesses'):
        xyz_file = os.path.join(f'{target_dir}/final_ts_guesses', file)
        output_file = os.path.join(f'{target_dir}/g16_input_files', f'{file.split(".xyz")[0]}.com')
        xyz_to_gaussian_input(xyz_file, output_file)


    #-jointly_optimize_reactants_and_products('[H:1]/[C:2](=[C:3](/[H:5])[O:6][H:7])[H:4].[O:8]=[C:9]([H:10])[H:11]', '[H:1][C:2]([C:3](=[O:6])[C:9]([O:8][H:7])([H:10])[H:11])([H:4])[H:5]')
    #-jointly_optimize_reactants_and_products('[C:1](=[O:2])([H:5])[H:6].[N:3]([H:4])([H:7])[H:8]', '[C:1]([O:2][N:3]([H:4])[H:8])([H:5])([H:6])[H:7]')
    #+jointly_optimize_reactants_and_products('[C:1](=[O:2])([H:5])[H:6].[C:3]([O:4][H:9])([H:7])([H:8])[H:10]','[C-:1]#[O+:2].[C:3]([O:4][H:6])([H:7])([H:8])[H:10].[H:5][H:9]')
    #+jointly_optimize_reactants_and_products('[C:1](=[O:2])([H:5])[H:6].[C:3]([O:4][H:9])([H:7])([H:8])[H:10]','[C:1]([O:2][H:7])([C:3]([O:4][H:9])([H:8])[H:10])([H:5])[H:6]')
    #+jointly_optimize_reactants_and_products('[C:1]([C:2](=[O:3])[H:6])([H:4])([H:5])[H:7]','[C:1](=[C:2](\[O:3][H:7])[H:6])(\[H:4])[H:5]')
    #+jointly_optimize_reactants_and_products('[C:1](=[C:2]([C:3](=[C:4](\[H:14])[H:15])\[H:16])/[H:7])(\[H:8])[H:9].[C:5](=[C:6](/[H:12])[H:13])(\[H:10])[H:11]','[C:1](=[C:2]=[C:3]([C:4]([C:5]([C:6]([H:7])([H:12])[H:13])([H:10])[H:11])([H:14])[H:15])[H:16])([H:8])[H:9]')
    #jointly_optimize_reactants_and_products('[H:1][C:2]([H:3])([C:4]([H:6])([H:7])[H:8])[H:5]', '[H:1]/[C:2]([H:3])=[C:4](/[H:7])[H:8].[H:5][H:6]')
    #-jointly_optimize_reactants_and_products('[C@@:1]1([Cl:4])([F:5])[O:2][C@:3]1([C@@:6]([F:8])([F:9])[F:10])[F:7]', '[C@@:1]([C:3](=[O:2])[F:7])([Cl:4])([F:5])[C@:6]([F:8])([F:9])[F:10]')
    #+jointly_optimize_reactants_and_products('[C:1](=[C:2](\[O:3][H:7])[H:6])(/[H:4])[H:5]', '[C:1]([C:2](=[O:3])[H:6])([H:4])([H:5])[H:7]')

    #jointly_optimize_reactants_and_products('[H:1][C:2]#[C:3][H:4].[H:5][H:6]','[H:1][C:2]([H:5])=[C:3]([H:6])[H:4]')
    #jointly_optimize_reactants_and_products('[C:1](=[O:2])([H:5])[H:6].[N:3]([H:4])([H:7])[H:8]','[C:1]([O:2][N:3]([H:4])[H:8])([H:5])([H:6])[H:7]')
    #jointly_optimize_reactants_and_products('[C-:1]#[O+:2].[H:3][H:4]','[C:1](=[O:2])([H:3])[H:4]')
    #jointly_optimize_reactants_and_products('[H:1]/[C:2](=[C:3](/[H:5])[O:6][H:7])[H:4].[O:8]=[C:9]([H:10])[H:11]','[H:1][C:2]([C:3](=[O:6])[C:9]([O:8][H:7])([H:10])[H:11])([H:4])[H:5]')
    #jointly_optimize_reactants_and_products('[H:8][C:1]#[N:2].[H:7][C:3]#[N:4].[C-:5]#[N:6]', '[C-:3]#[N:4].[C:1](=[N:2][H:7])([C:5]#[N:6])[H:8]', solvent='water')
    #jointly_optimize_reactants_and_products('[H:1]/[C:2](=[C:3](/[H:5])[O:6][H:7])[H:4].[O:8]=[C:9]([H:10])[H:11]','[H:1]/[C:2](=[C:3](/[H:5])[O:6][C:9]([O:8][H:7])([H:10])[H:11])[H:4]')
    #jointly_optimize_reactants_and_products('[H]C#N.[H]C#N.[H]C#N', '[H:9][C:3]#[N:4].[C:1](=[N:2][H:8])([C:5]#[N:6])[H:7]')
    #aligned_molecule = align_fragment_with_molecule('C[C@@H](N)C(=O)O', 'CC(C)(C)C[C@@H](NC(=O)[C@@H](NC(=O)C)CC1=CC=CC=C1)C(=O)O')
