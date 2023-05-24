from reaction_profile_generator.generator import find_ts_guess

from rdkit import Chem
from rdkit.Chem import AllChem


def get_smiles_strings(filename):
    ''' a function that opens a file, reads in every line as a reaction smiles and returns them as a list. '''
    with open(filename, 'r') as f:
        lines = f.readlines()
        smiles_strings = [line.rstrip().split()[-1] for line in lines]
    return smiles_strings

def get_ts_guess(reaction_smiles):
    ''' a function that splits up a reaction smiles in reactant and product, and then calls the function find_ts_guess with these as parameters. '''
    reactant, product = reaction_smiles.split('>>')
    ts_guess = find_ts_guess(reactant, product)
    return ts_guess

def confirm_ts_nature(ts_guess):
    pass


if __name__ == "__main__":
    smiles_strings = get_smiles_strings('reactions_am.txt')
    for smiles_string in smiles_strings[16:]:
        ts_guess = get_ts_guess(smiles_string)
        print(smiles_string, '\t', ts_guess)



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
