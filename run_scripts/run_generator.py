from reaction_profile_generator.generator import jointly_optimize_reactants_and_products

from rdkit import Chem
from rdkit.Chem import AllChem


if __name__ == "__main__":
    #jointly_optimize_reactants_and_products('[H:1][C:2]#[C:3][H:4].[H:5][H:6]','[H:1][C:2]([H:5])=[C:3]([H:6])[H:4]')
    #jointly_optimize_reactants_and_products('[C:1](=[O:2])([H:5])[H:6].[N:3]([H:4])([H:7])[H:8]','[C:1]([O:2][N:3]([H:4])[H:8])([H:5])([H:6])[H:7]')
    #jointly_optimize_reactants_and_products('[C-:1]#[O+:2].[H:3][H:4]','[C:1](=[O:2])([H:3])[H:4]')
    #jointly_optimize_reactants_and_products('[H:1]/[C:2](=[C:3](/[H:5])[O:6][H:7])[H:4].[O:8]=[C:9]([H:10])[H:11]','[H:1][C:2]([C:3](=[O:6])[C:9]([O:8][H:7])([H:10])[H:11])([H:4])[H:5]')
    #jointly_optimize_reactants_and_products('[H:8][C:1]#[N:2].[H:7][C:3]#[N:4].[C-:5]#[N:6]', '[C-:3]#[N:4].[C:1](=[N:2][H:7])([C:5]#[N:6])[H:8]')
    jointly_optimize_reactants_and_products('[H:1]/[C:2](=[C:3](/[H:5])[O:6][H:7])[H:4].[O:8]=[C:9]([H:10])[H:11]', '[H:1]/[C:2](=[C:3](/[H:5])[O:6][H:11])[H:4].[H:7][C:9](=[O:8])[H:10]')
    #jointly_optimize_reactants_and_products('[H]C#N.[H]C#N.[H]C#N', '[H:9][C:3]#[N:4].[C:1](=[N:2][H:8])([C:5]#[N:6])[H:7]')
    #aligned_molecule = align_fragment_with_molecule('C[C@@H](N)C(=O)O', 'CC(C)(C)C[C@@H](NC(=O)[C@@H](NC(=O)C)CC1=CC=CC=C1)C(=O)O')
