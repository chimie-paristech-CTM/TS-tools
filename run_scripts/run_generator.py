from reaction_profile_generator.generator import jointly_optimize_reactants_and_products

from rdkit import Chem
from rdkit.Chem import AllChem


if __name__ == "__main__":
    #-jointly_optimize_reactants_and_products('[H:1]/[C:2](=[C:3](/[H:5])[O:6][H:7])[H:4].[O:8]=[C:9]([H:10])[H:11]', '[H:1][C:2]([C:3](=[O:6])[C:9]([O:8][H:7])([H:10])[H:11])([H:4])[H:5]')
    #-jointly_optimize_reactants_and_products('[C:1](=[O:2])([H:5])[H:6].[N:3]([H:4])([H:7])[H:8]', '[C:1]([O:2][N:3]([H:4])[H:8])([H:5])([H:6])[H:7]')
    #+jointly_optimize_reactants_and_products('[C:1](=[O:2])([H:5])[H:6].[C:3]([O:4][H:9])([H:7])([H:8])[H:10]','[C-:1]#[O+:2].[C:3]([O:4][H:6])([H:7])([H:8])[H:10].[H:5][H:9]')
    #+jointly_optimize_reactants_and_products('[C:1](=[O:2])([H:5])[H:6].[C:3]([O:4][H:9])([H:7])([H:8])[H:10]','[C:1]([O:2][H:7])([C:3]([O:4][H:9])([H:8])[H:10])([H:5])[H:6]')
    #+jointly_optimize_reactants_and_products('[C:1]([C:2](=[O:3])[H:6])([H:4])([H:5])[H:7]','[C:1](=[C:2](\[O:3][H:7])[H:6])(\[H:4])[H:5]')
    #+jointly_optimize_reactants_and_products('[C:1](=[C:2]([C:3](=[C:4](\[H:14])[H:15])\[H:16])/[H:7])(\[H:8])[H:9].[C:5](=[C:6](/[H:12])[H:13])(\[H:10])[H:11]','[C:1](=[C:2]=[C:3]([C:4]([C:5]([C:6]([H:7])([H:12])[H:13])([H:10])[H:11])([H:14])[H:15])[H:16])([H:8])[H:9]')
    #?jointly_optimize_reactants_and_products('[H:1][C:2]([H:3])([C:4]([H:6])([H:7])[H:8])[H:5]', '[H:1]/[C:2]([H:3])=[C:4](/[H:7])[H:8].[H:5][H:6]')
    jointly_optimize_reactants_and_products('[C@@:1]1([Cl:4])([F:5])[O:2][C@:3]1([C@@:6]([F:8])([F:9])[F:10])[F:7]', '[C@@:1]([C:3](=[O:2])[F:7])([Cl:4])([F:5])[C@:6]([F:8])([F:9])[F:10]')


    #jointly_optimize_reactants_and_products('[H:1][C:2]#[C:3][H:4].[H:5][H:6]','[H:1][C:2]([H:5])=[C:3]([H:6])[H:4]')
    #jointly_optimize_reactants_and_products('[C:1](=[O:2])([H:5])[H:6].[N:3]([H:4])([H:7])[H:8]','[C:1]([O:2][N:3]([H:4])[H:8])([H:5])([H:6])[H:7]')
    #jointly_optimize_reactants_and_products('[C-:1]#[O+:2].[H:3][H:4]','[C:1](=[O:2])([H:3])[H:4]')
    #jointly_optimize_reactants_and_products('[H:1]/[C:2](=[C:3](/[H:5])[O:6][H:7])[H:4].[O:8]=[C:9]([H:10])[H:11]','[H:1][C:2]([C:3](=[O:6])[C:9]([O:8][H:7])([H:10])[H:11])([H:4])[H:5]')
    #jointly_optimize_reactants_and_products('[H:8][C:1]#[N:2].[H:7][C:3]#[N:4].[C-:5]#[N:6]', '[C-:3]#[N:4].[C:1](=[N:2][H:7])([C:5]#[N:6])[H:8]', solvent='water')
    #jointly_optimize_reactants_and_products('[H:1]/[C:2](=[C:3](/[H:5])[O:6][H:7])[H:4].[O:8]=[C:9]([H:10])[H:11]','[H:1]/[C:2](=[C:3](/[H:5])[O:6][C:9]([O:8][H:7])([H:10])[H:11])[H:4]')
    #jointly_optimize_reactants_and_products('[H]C#N.[H]C#N.[H]C#N', '[H:9][C:3]#[N:4].[C:1](=[N:2][H:8])([C:5]#[N:6])[H:7]')
    #aligned_molecule = align_fragment_with_molecule('C[C@@H](N)C(=O)O', 'CC(C)(C)C[C@@H](NC(=O)[C@@H](NC(=O)C)CC1=CC=CC=C1)C(=O)O')
