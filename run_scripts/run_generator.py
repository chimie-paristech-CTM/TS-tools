from reaction_profile_generator.generator import jointly_optimize_reactants_and_products

from rdkit import Chem
from rdkit.Chem import AllChem


if __name__ == "__main__":
    jointly_optimize_reactants_and_products('[H]C#N.[H]C#N.[C-]#N', '[C-:3]#[N:4].[C:1](=[N:2][H:8])([C:5]#[N:6])[H:7]')
    #jointly_optimize_reactants_and_products('[H]C#N.[H]C#N.[H]C#N', '[H:9][C:3]#[N:4].[C:1](=[N:2][H:8])([C:5]#[N:6])[H:7]')
    #aligned_molecule = align_fragment_with_molecule('C[C@@H](N)C(=O)O', 'CC(C)(C)C[C@@H](NC(=O)[C@@H](NC(=O)C)CC1=CC=CC=C1)C(=O)O')
