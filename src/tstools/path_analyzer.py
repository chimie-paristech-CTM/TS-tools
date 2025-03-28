import numpy as np
import autode as ade
import os
from scipy.spatial import distance_matrix
from collections import defaultdict
import subprocess
from rdkit import Chem

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from rdkit.Chem import rdDetermineBonds
from autode.mol_graphs import make_graph
import re
import warnings

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*') 

from tstools.confirm_ts_guess import validate_ts_guess

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.decomposition._pca")

hartree_to_kcal = 627.509474

ps = Chem.SmilesParserParams()
ps.removeHs = False

class PathAnalyzer:

    def __init__(self, path, energies, potentials, path_xyz_files, proc):
        """
        Initialize a PathAnalyzer object.

        Returns:
        None
        """
        self.path = path
        self.energies = energies
        self.potentials = potentials
        self.path_xyz_files = path_xyz_files

        self.true_energies = self.get_true_energies()
        self.reactant_mol, self.product_mol = self.get_reactant_and_product_mol()
        self.points_of_interest, self.pca_coord = self.analyze_reaction_path()

        self.barrier_estimate = None
        self.proc = proc

    def check_if_reactive_path_is_reasonable(self):
        """
        Checks if the proposed reactive path between reactants and products is reasonable by examining bond formation 
        and breaking at the most asynchronous point along the reaction path.

        Returns:
            bool: True if the reactive path is consistent with expected bond changes, False otherwise.

        Method:
            1. Identifies the bonds formed and broken during the reaction based on the difference in graph edges (bonds)
                between the reactant and product molecules.
            2. Calculates the bonds already formed in the extreme point molecule with a low tolerance, and checks if they 
                are a subset of the expected formed bonds.
            3. Calculates the bonds already broken in the extreme point molecule with a high tolerance, and checks if they 
                are a subset of the expected broken bonds.
            4. Returns True if both conditions hold; otherwise, returns False.
        """
        extreme_point_mol = ade.Molecule(self.path_xyz_files[self.points_of_interest[0]])

        bonds_formed = set(self.product_mol.graph.edges()) - set(self.reactant_mol.graph.edges())
        bonds_broken = set(self.reactant_mol.graph.edges()) - set(self.product_mol.graph.edges())
        active_bonds = bonds_formed.union(bonds_broken)
  
        make_graph(extreme_point_mol, rel_tolerance=0.1, allow_invalid_valancies=True)
        bonds_already_formed = set(extreme_point_mol.graph.edges()) - set(self.reactant_mol.graph.edges())
        if not bonds_already_formed.issubset(active_bonds):
            return False

        make_graph(extreme_point_mol, rel_tolerance=0.5, allow_invalid_valancies=True)
        bonds_already_broken = set(self.reactant_mol.graph.edges()) - set(extreme_point_mol.graph.edges())
        if not bonds_already_broken.issubset(active_bonds):
            return False
        
        return True

    def check_for_potential_intermediates(self):
        """
        Identifies and validates potential intermediates along the reaction path.

        Returns:
            str or None: SMILES string of the validated intermediate, or None if no intermediate is found or the 
            candidate structure reverts to reactant or product after optimization.

        Method:
            1. Checks if the extreme point molecule has a bonding pattern different from both reactants and products.
            2. Identifies isolated subgraphs of atoms that differ from reactants and products, representing active regions.
            3. Filters and optimizes the intermediate candidate structure for validation.
            4. Verifies the optimized candidate to ensure it does not collapse to a reactant or product structure.
            5. Converts the intermediate structure to a SMILES string with atom mapping for easy identification.

        Raises:
            None: Errors are handled by returning None if an issue arises with the intermediate candidate.
        """
        pot_inter_mol = ade.Molecule(self.path_xyz_files[self.points_of_interest[0]])

        if set(pot_inter_mol.graph.edges) != set(self.reactant_mol.graph.edges) and \
            set(pot_inter_mol.graph.edges) != set(self.product_mol.graph.edges):
            subgraphs_intermediate = identify_isolated_subgraphs(pot_inter_mol.graph.edges, len(pot_inter_mol.atoms))
            subgraphs_reactant = identify_isolated_subgraphs(self.reactant_mol.graph.edges, len(self.reactant_mol.atoms))
            subgraphs_product = identify_isolated_subgraphs(self.product_mol.graph.edges, len(self.product_mol.atoms))

            # filter out molecules that have not yet changed compared to the reactants and products
            active_subgraphs = [s for s in subgraphs_intermediate \
                                if s not in subgraphs_reactant and s not in subgraphs_product]
            try:
                active_atoms = set.union(*active_subgraphs)
            except:
                return None

            filter_xyz(self.path_xyz_files[self.points_of_interest[0]], 'candidate_intermediate.xyz', active_atoms)
            optimize_candidate_intermediate(self.path, 'candidate_intermediate.xyz') 
            
            # double check that the intermediate has not collapsed to the (truncated) reactant or product
            filter_xyz(os.path.join(self.path.rp_geometries_dir, 'reactants_geometry.xyz'), 'truncated_reactants.xyz', active_atoms)
            filter_xyz(os.path.join(self.path.rp_geometries_dir, 'products_geometry.xyz'), 'truncated_products.xyz', active_atoms)

            truncated_reactant_mol = ade.Molecule('truncated_reactants.xyz')
            truncated_product_mol = ade.Molecule('truncated_products.xyz')
            pot_inter_mol_opt = ade.Molecule('candidate_intermediate_opt.xyz')

            if set(pot_inter_mol_opt.graph.edges) != set(truncated_reactant_mol.graph.edges) and \
                set(pot_inter_mol_opt.graph.edges) != set(truncated_product_mol.graph.edges): 
                raw_pot_inter_rdkit_mol = Chem.MolFromXYZFile('candidate_intermediate_opt.xyz')
                pot_inter_rdkit_mol = Chem.Mol(raw_pot_inter_rdkit_mol)

                rdDetermineBonds.DetermineBonds(pot_inter_rdkit_mol)

                # If molecules that also appear in reactant or product are recovered, then the intermediate is unproductive
                product_mol = Chem.MolFromSmiles(self.path.product_smiles)
                [atom.SetAtomMapNum(0) for atom in product_mol.GetAtoms()]
                unmapped_product_smiles = Chem.MolToSmiles(Chem.RemoveHs(product_mol))  

                reactant_mol = Chem.MolFromSmiles(self.path.reactant_smiles)
                [atom.SetAtomMapNum(0) for atom in reactant_mol.GetAtoms()]
                unmapped_reactant_smiles = Chem.MolToSmiles(Chem.RemoveHs(reactant_mol))

                unmapped_intermediate_smiles = Chem.MolToSmiles(Chem.RemoveHs(pot_inter_rdkit_mol))

                if set(unmapped_intermediate_smiles.split('.')).isdisjoint(set(unmapped_reactant_smiles.split('.'))) \
                    and set(unmapped_intermediate_smiles.split('.')).isdisjoint(set(unmapped_product_smiles.split('.'))):

                    for idx, atom in zip(active_atoms, [atom for atom in pot_inter_rdkit_mol.GetAtoms()]):
                        atom.SetAtomMapNum(self.path.atom_idx_dict[idx])

                    mapped_intermediate_smiles = Chem.MolToSmiles(pot_inter_rdkit_mol)
                    return mapped_intermediate_smiles

        return None
        
    def select_ts_guesses(self, reaction_dir, freq_cut_off, charge, multiplicity, solvent):
        """
        Ranks and filters transition state (TS) guesses based on energy, structural similarity, and vibrational analysis.

        Args:
            reaction_dir (str): Path to the reaction directory for storing TS validation outputs.
            freq_cut_off (float): Cutoff value for the imaginary frequency that identifies a TS.
            charge (int): Total charge of the system.
            multiplicity (int): Spin multiplicity of the system.
            solvent (str): Solvent model to be used during TS validation.

        Returns:
            list of str: Filenames of the final list of valid TS guess structures that meet all criteria.
        """        
        # first rerank based on the energy
        ts_guess_dict = {index : self.true_energies[index] for index in self.points_of_interest}
        sorted_items = sorted(ts_guess_dict.items(), key=lambda item: item[1], reverse=True)

        energy_sorted_indices_most_asynchronous = [item[0] for item in sorted_items if item[0] in self.points_of_interest[:10]]
        energy_sorted_indices_all = [item[0] for item in sorted_items if item[0] in self.points_of_interest]

        # take the highest energy point among the 10 most asynchronous ones as the first candidate guess,
        # then iterate through all the remaining indices and remove guess geometries that are 
        # too similar (based on the pca coordinates)
        similarity_filtered_indices = [energy_sorted_indices_most_asynchronous[0]]
        for index in energy_sorted_indices_all[1:]:
            if any([compute_euclidean_distance(self.pca_coord[i], self.pca_coord[index]) 
                    < 0.1 for i in similarity_filtered_indices]):
                continue
            else:
                similarity_filtered_indices.append(index)
                if len(similarity_filtered_indices) == 4:
                    break

        validated_indices = [i for i in similarity_filtered_indices if validate_ts_guess(
                    self.path_xyz_files[i], reaction_dir, freq_cut_off, charge, multiplicity, solvent) == True]

        guesses_list = [self.path_xyz_files[i] for i in validated_indices]

        # set the barrier estimate attribute if guesses list is not empty
        if len(guesses_list) > 0:
            self.barrier_estimate = (self.true_energies[validated_indices[0]] - self.true_energies[0]) * hartree_to_kcal

        return guesses_list

    def get_true_energies(self):
        """
        Computes the true energies by applying potential energy corrections to the initial energy values.

        Returns:
            list of float or None: A list of true energies with corrections applied, or None if `self.energies` is None.
        """
        if self.energies is not None:
            true_energies = list(np.array(self.energies) - np.array(self.potentials))
        else:
            true_energies = None
        
        return true_energies

    def analyze_reaction_path(self):
        """
        Analyzes the reaction path by examining synchronous and asynchronous components of bond displacement vectors.

        Returns:
            tuple: A tuple containing:
                - list of int: Indices of the points with the highest asynchronous character.
                - list of dict: Bond displacement vectors for each point along the path.

        Method:
            1. **Bond Length Dictionaries:** Uses `get_bond_length_dicts` to obtain dictionaries for bond formation 
                and breaking lengths.
            2. **Bond Displacement Vectors:** Calculates bond displacement vectors for each point along the path, 
                normalizing them using Min-Max scaling.
            3. **Synchronous Component:** Defines the synchronous path component as the vector from the first to the 
                last point and projects bond displacement vectors onto it.
            4. **Explained Variance (Synchronous):** Computes the variance in projections onto the synchronous component.
            5. **Residual Analysis (Asynchronous):** Removes synchronous component variation, then applies PCA to capture 
                the main asynchronous direction.
            6. **Variance Calculation:** Calculates total variance and computes explained variance ratios for both 
                synchronous and asynchronous components. Issues a warning if the synchronous component explains 
                insufficient variance.
            7. **Scaling and Normalization:** Rescales synchronous and asynchronous projections using Min-Max scaling.
            8. **Asynchronous Points Identification:** Determines the most asynchronous points by analyzing extreme values 
                in asynchronous projections or central points in the synchronous trajectory if asynchronicity is limited.
        """
        # step 1: get the bond length dicts
        bond_vector_list = []

        bonds_formed_length_dict, bonds_broken_length_dict = self.get_bond_length_dicts()

        # step 2: extract the bond displacement vectors for every point along the path
        for file in self.path_xyz_files:
            dist_mat = get_distance_matrix(file)
            bond_vector = extract_active_bond_vector(dist_mat, bonds_formed_length_dict, bonds_broken_length_dict)
            bond_vector_list.append(bond_vector)

        bond_vector_array = np.array(bond_vector_list)
        bond_vector_scaler = MinMaxScaler()
        bond_vector_array = bond_vector_scaler.fit_transform(bond_vector_array)

        # Step 3: Define the synchronous component as the direction from the first to the last point
        first_point = bond_vector_array[0]
        last_point = bond_vector_array[-1]

        synchronous_component = last_point - first_point
        synchronous_component /= np.linalg.norm(synchronous_component)  # Normalize the vector

        # step 4: project all displacement vectors onto the synchronous component
        projections_synchronous = np.dot(bond_vector_array - first_point, synchronous_component)

        # Variance explained by the synchronous part of the path
        explained_var_syn = np.var(projections_synchronous)

        # step 5: Remove the variation along the synchronous component
        projections_synchronous_array = projections_synchronous[:, np.newaxis] * synchronous_component
        residuals = bond_vector_array - projections_synchronous_array

        # step 6: Apply PCA to the residuals to capture the principal asynchronous component
        pca = PCA(n_components=1)  # try a single additional component
        projections_asynchronous = pca.fit_transform(residuals).flatten() 

        # Variance explained by the asynchronous component
        explained_var_asyn = pca.explained_variance_[0]

        # step 7: Calculate the total variance in the data
        total_variance = np.var(bond_vector_array, axis=0).sum()

        # step 8: Calculate the explained variance ratio for each component
        explained_var_ratio_syn = explained_var_syn / total_variance
        explained_var_ratio_asyn = explained_var_asyn / total_variance
        total_explained_var = explained_var_ratio_syn + explained_var_ratio_asyn

        if total_explained_var < 0.95 or explained_var_ratio_syn < 0.6:
            print(f'WARNING: reactive path for {self.path.rxn_id} exhibits atypical behavior:') 
            print(f'total explained variance amounts to {total_explained_var}, explained variance by synchronous path amounts to {explained_var_ratio_syn}')

        # scale the individual components
        synchronous_scaler = MinMaxScaler()
        projections_synchronous = np.array(synchronous_scaler.fit_transform(projections_synchronous.reshape(-1,1))).reshape(-1)

        asynchronous_scaler = MinMaxScaler()
        projections_asynchronous = asynchronous_scaler.fit_transform(projections_asynchronous.reshape(-1,1))
        projections_asynchronous = np.array(projections_asynchronous - projections_asynchronous[0]).reshape(-1)

        # step 9: get the points that are the most asynchronous
        top_indices = []
        if explained_var_ratio_syn >= 0.97: # limited to no asynchronicity 
            interval = (projections_synchronous[-1] - projections_synchronous[0])
            for i, syn_value in enumerate(projections_synchronous):
                if syn_value > projections_synchronous[0] + interval * 0.2 \
                        and syn_value < projections_synchronous[-1] - interval * 0.2:
                    top_indices.append(i)
        else:
            if abs(min(projections_asynchronous)) > abs(max(projections_asynchronous)): # excursion across negative side
                indices_to_retain = [i for i, p in enumerate(projections_asynchronous) if p < 0.65 * min(projections_asynchronous)]  
                top_indices = sorted(indices_to_retain, key=lambda i: projections_asynchronous[i], reverse=False)   
            else:
                indices_to_retain = [i for i, p in enumerate(projections_asynchronous) if p > 0.65 * max(projections_asynchronous)]  
                top_indices = sorted(indices_to_retain, key=lambda i: projections_asynchronous[i], reverse=True)  

        pca_coord = np.array(list(zip(projections_synchronous, projections_asynchronous)))

        return top_indices, pca_coord
    
    def get_bond_length_dicts(self):
        """
        Constructs dictionaries of bond lengths for bonds formed and broken between reactants and products.

        Returns:
            tuple: A tuple containing two dictionaries:
                - bonds_formed_length_dict (dict): Bond lengths for bonds formed in the reaction, 
                    with bond pairs as keys and bond lengths as values.
                - bonds_broken_length_dict (dict): Bond lengths for bonds broken in the reaction, 
                    with bond pairs as keys and bond lengths as values.
        """
        reactant_dist_mat, product_dist_mat = self.get_reactant_and_product_dist_mat()

        bonds_formed = list(set(self.product_mol.graph.edges()) - set(self.reactant_mol.graph.edges()))
        bonds_broken = list(set(self.reactant_mol.graph.edges()) - set(self.product_mol.graph.edges()))

        bonds_formed_length_dict = {}
        bonds_broken_length_dict = {}

        for bond in bonds_formed:
            bonds_formed_length_dict[bond] = product_dist_mat[bond[0]][bond[1]]

        for bond in bonds_broken:
            bonds_broken_length_dict[bond] = reactant_dist_mat[bond[0]][bond[1]]

        return bonds_formed_length_dict, bonds_broken_length_dict

    def get_reactant_and_product_mol(self):
        """
        Loads and returns the reactant and product molecules from their respective XYZ geometry files.

        Returns:
            tuple: A tuple containing:
                - reactant_mol (ade.Molecule): An ADE molecule object for the reactant, loaded from the 
                    'reactants_geometry.xyz' file.
                - product_mol (ade.Molecule): An ADE molecule object for the product, loaded from the 
                    'products_geometry.xyz' file.
        """
        reactant_xyz_path = os.path.join(self.path.rp_geometries_dir, 'reactants_geometry.xyz')
        product_xyz_path = os.path.join(self.path.rp_geometries_dir, 'products_geometry.xyz')

        reactant_mol = ade.Molecule(reactant_xyz_path)
        product_mol = ade.Molecule(product_xyz_path)

        return reactant_mol, product_mol
    
    def get_reactant_and_product_dist_mat(self):
        """
        Computes and returns the distance matrices for the reactant and product molecules.

        Returns:
            tuple: A tuple containing:
                - reactant_dist_mat (numpy.ndarray): A 2D array representing the pairwise atomic distances 
                    for the reactant molecule.
                - product_dist_mat (numpy.ndarray): A 2D array representing the pairwise atomic distances 
                    for the product molecule.
        """
        reactant_xyz_path = os.path.join(self.path.rp_geometries_dir, 'reactants_geometry.xyz')
        product_xyz_path = os.path.join(self.path.rp_geometries_dir, 'products_geometry.xyz')

        reactant_dist_mat = get_distance_matrix(reactant_xyz_path)
        product_dist_mat = get_distance_matrix(product_xyz_path)

        return reactant_dist_mat, product_dist_mat
    
    def generate_stepwise_reaction_smiles(self, mapped_intermediate_smiles):
        """
        Generates stepwise reaction SMILES for a reaction, including intermediates and unaffected molecules.

        Args:
            mapped_intermediate_smiles (str): The SMILES string representing the intermediate molecule that will 
                                           be used to form the stepwise reaction pathway.

        Returns:
            tuple: A tuple containing two SMILES strings:
                - reaction_smiles1 (str): The first step of the reaction in SMILES format, from reactants to the intermediate.
                - reaction_smiles2 (str): The second step of the reaction in SMILES format, from the intermediate to the products.

        Method:
            1. Identifies the unaffected atom set by comparing the reactant and intermediate SMILES using atom maps.
            2. Splits the reactant SMILES into individual molecules and categorizes them into unaffected and affected based 
                on atom map numbers.
            3. Constructs the first reaction SMILES from the affected molecules to the intermediate.
            4. If there are unaffected molecules:
                - For non-catalytic reactions, constructs the second reaction SMILES from the intermediate to the product.
                - For autocatalytic reactions, attempts to modify the product SMILES based on the unaffected atom set.
            5. Returns both reaction SMILES strings.
        """
        unaffected_atom_set = find_unique_atom_maps(self.path.reactant_smiles, mapped_intermediate_smiles)
        affected_molecules, unaffected_molecules = [], []

        for molecule_smiles in self.path.reactant_smiles.split('.'):
            if extract_atom_map_numbers(molecule_smiles).issubset(unaffected_atom_set):
                unaffected_molecules.append(molecule_smiles)
            else:
                affected_molecules.append(molecule_smiles)

        reaction_smiles1 = f'{".".join(affected_molecules)}>>{mapped_intermediate_smiles}'
        if len(unaffected_molecules) != 0:
            if not self.reaction_is_auto_or_solvent_catalytic():
                reaction_smiles2 = f'{mapped_intermediate_smiles}.{".".join(unaffected_molecules)}>>{self.path.product_smiles}'
            else:
                try:
                    modified_product_smiles = self.filter_product_smiles(list(unaffected_atom_set))
                    if mapped_intermediate_smiles != modified_product_smiles:
                        reaction_smiles2 = f'{mapped_intermediate_smiles}>>{modified_product_smiles}'
                    else:
                        reaction_smiles2 = None
                except:
                    print('Generation of second reaction SMILES for autocatalytic reaction failed')
                    reaction_smiles2 = f'{mapped_intermediate_smiles}.{".".join(unaffected_molecules)}>>{self.path.product_smiles}' # if the makeshift code below doesn't work, then just continue with the extended SMILES
        else:
            reaction_smiles2 = f'{mapped_intermediate_smiles}>>{self.path.product_smiles}'

        return reaction_smiles1, reaction_smiles2

    def reaction_is_auto_or_solvent_catalytic(self):
        """
        Determines if a reaction is autocatalytic or solvent-catalytic based on the presence of overlapping molecules 
        between the reactant and product sides.

        Returns:
            bool: `True` if the reaction is autocatalytic or solvent-catalytic (i.e., a molecule is present in both 
                the reactant and product SMILES), otherwise `False`.

        Method:
            1. Converts the reactant and product SMILES to molecule objects using RDKit.
            2. Removes hydrogen atoms and atom map numbers from the molecules for comparison.
            3. Checks if any molecules appear in both the reactant and product sides by splitting the SMILES strings 
                and checking for intersection.
            4. Returns `True` if there is any overlap between reactant and product molecules, indicating an autocatalytic 
                or solvent-catalytic reaction, otherwise `False`.
        """
        # a reaction is auto or solvent catalytic if a molecule appears both on reactant and product side
        product_mol = Chem.MolFromSmiles(self.path.product_smiles)
        [atom.SetAtomMapNum(0) for atom in product_mol.GetAtoms()]
        unmapped_product_smiles = Chem.MolToSmiles(Chem.RemoveHs(product_mol)) 

        reactant_mol = Chem.MolFromSmiles(self.path.reactant_smiles)
        [atom.SetAtomMapNum(0) for atom in reactant_mol.GetAtoms()]
        unmapped_reactant_smiles = Chem.MolToSmiles(Chem.RemoveHs(reactant_mol)) 

        # check if there is overlap between the molecules on reactant and product side
        if set(unmapped_product_smiles.split('.')).intersection(set(unmapped_reactant_smiles.split('.'))):
            return True
        else:
            return False

    # TODO: this is also a very makeshift solution... -> needs fixing  
    def filter_product_smiles(self, unaffected_atom_list):
        """
        Filters the product SMILES by removing molecules that contain atoms listed in the unaffected atom list, 
        and reassigns atom map numbers to ensure the retained atoms' identities are preserved.

        Args:
            unaffected_atom_list (list): A list of atom map numbers representing the atoms that should not be 
                                        altered during the filtering process.

        Returns:
            str: The modified product SMILES string with the unaffected atoms removed and reassignments applied 
                to the remaining atoms, ensuring atom map numbers are retained where possible.
        """
        molecules_to_retain = []
        molecules_with_deleted_atoms = []

        for molecule in self.path.product_smiles.split('.'):
            if any([(unaffected_atom in molecule) for unaffected_atom in unaffected_atom_list]):
                molecules_with_deleted_atoms.append(molecule)
            else:
                molecules_to_retain.append(molecule)

        mol = Chem.MolFromSmiles('.'.join(molecules_with_deleted_atoms).strip('.'), ps)
        atom_indices_to_delete, all_involved_atoms = [], []
        for atom in mol.GetAtoms():
            if str(atom.GetAtomMapNum()) in unaffected_atom_list:
                atom_indices_to_delete.append(atom.GetIdx())
                for neighbor in atom.GetNeighbors():
                    all_involved_atoms.append(neighbor.GetIdx())

        remaining_atoms = list(set(all_involved_atoms) - set(atom_indices_to_delete))
        
        # Only when length is two can you safely finalize the SMILES string
        if len(remaining_atoms) != 2:
            return None
        else:
            # Create an editable molecule object and make bond + remove atoms to be deleted
            editable_mol = Chem.RWMol(mol)
            editable_mol.AddBond(remaining_atoms[0], remaining_atoms[1], order=Chem.rdchem.BondType.SINGLE)
            for atom_idx in sorted(atom_indices_to_delete, reverse=True):
                editable_mol.RemoveAtom(atom_idx)

        return Chem.MolToSmiles(editable_mol)


def extract_atom_map_numbers(smiles):
    """    
    Extracts atom map numbers from a given SMILES string.

    Args:
        smiles (str): The SMILES representation of a molecule, possibly containing atom map numbers.

    Returns:
        set: A set of atom map numbers extracted from the SMILES string. The atom map numbers are returned as strings 
              for easy comparison and manipulation.
    """
    # Regular expression to match atom maps like :1, :2, :3, etc.
    atom_maps = re.findall(r':(\d+)', smiles)
    return set(atom_maps)  # return as a set for easy comparison

def find_unique_atom_maps(smiles1, smiles2):
    """    
    Finds atom map numbers that are unique to the first SMILES string when compared to the second.

    Args:
        smiles1 (str): The first SMILES string, potentially containing atom map numbers.
        smiles2 (str): The second SMILES string, potentially containing atom map numbers.

    Returns:
        set: A set of atom map numbers that are unique to the first SMILES string, i.e., present in `smiles1` 
             but not in `smiles2`.
    """
    atom_maps1 = extract_atom_map_numbers(smiles1)
    atom_maps2 = extract_atom_map_numbers(smiles2)
    
    # Find atom map numbers that are in smiles1 but not in smiles2
    unique_to_smiles1 = atom_maps1 - atom_maps2
    
    return unique_to_smiles1


def get_distance_matrix(filename):
    """    
    Computes the distance matrix for the atoms in a given geometry file.

    Args:
        filename (str): Path to the file containing the atomic coordinates in XYZ format.

    Returns:
        numpy.ndarray: A 2D array (distance matrix) where each element [i, j] represents the Euclidean distance 
                        between atom i and atom j in the geometry.
    """
    geometry = []
    with open(filename, 'r') as file:
        lines = file.readlines()

        for line in lines[2:]:
            parts = line.split()
            geometry.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    geometry = np.array(geometry)

    return distance_matrix(geometry, geometry)


def extract_active_bond_vector(dist_matrix, active_bond_dict_formed, active_bond_dict_broken):
    """    
    Extracts a vector representing the changes in bond lengths for active bonds formed and broken along a reaction path.

    Args:
        dist_matrix (numpy.ndarray): A 2D array representing the distance matrix of atomic coordinates. 
                                      Each element [i, j] represents the distance between atom i and atom j.
        active_bond_dict_formed (dict): A dictionary where keys are tuples of atom indices representing bonds 
                                        that are formed in the reaction, and values are the initial bond lengths.
        active_bond_dict_broken (dict): A dictionary where keys are tuples of atom indices representing bonds 
                                        that are broken in the reaction, and values are the initial bond lengths.

    Returns:
        numpy.ndarray: A vector (1D array) where each element corresponds to the change in bond length for 
                       an active bond (either formed or broken). Positive values indicate lengthening, 
                       and negative values indicate shortening of the bonds.
    """
    bond_length_vector = []

    for bond in active_bond_dict_formed.keys():
        bond_length = dist_matrix[bond[0]][bond[1]]
        bond_length_vector.append(bond_length - active_bond_dict_formed[bond])

    for bond in active_bond_dict_broken.keys():
        bond_length = dist_matrix[bond[0]][bond[1]]
        bond_length_vector.append(bond_length - active_bond_dict_broken[bond])

    return np.array(bond_length_vector)


def compute_euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def identify_isolated_subgraphs(edges, n_atoms):
    """    
    Identifies isolated subgraphs in a molecular graph based on the provided edges and number of atoms.

    This function constructs a graph using the provided bond edges and then finds the connected components 
    (subgraphs) using depth-first search (DFS). It returns the connected subgraphs along with isolated atoms 
    that are not connected to any other atoms, treating each isolated atom as a separate subgraph.

    Args:
        edges (list of tuples): A list of tuples where each tuple (i, j) represents a bond between atom i and atom j.
        n_atoms (int): The total number of atoms in the molecule.

    Returns:
        list of sets: A list of sets, where each set represents a connected subgraph. If an atom is isolated (i.e., 
                       not part of any bonds), it will be included in its own set as a single-node subgraph.
    """
    # Step 1: Build the adjacency list
    graph = defaultdict(set)
    for edge in edges:
        node1, node2 = edge
        graph[node1].add(node2)
        graph[node2].add(node1)
    
    # Step 2: Find connected components using DFS
    def dfs(node, visited, component):
        stack = [node]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                component.add(current)
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)
    
    visited = set()
    subgraphs = []
    
    for node in graph:
        if node not in visited:
            component = set()
            dfs(node, visited, component)
            subgraphs.append(component)
    
    # Extracting all nodes that are part of any component
    all_nodes_in_components = set()
    for component in subgraphs:
        all_nodes_in_components.update(component)


    for i in range(n_atoms):
        if i not in all_nodes_in_components:
            subgraphs.append(set([i]))

    return subgraphs


def filter_xyz(input_xyz_file, output_xyz_file, indices_to_retain):
    """
    Filters an XYZ file to retain only the atoms at specified indices.
    
    Parameters:
    - input_xyz_file: Path to the input XYZ file.
    - output_xyz_file: Path to save the filtered XYZ file.
    - indices_to_retain: A set of indices (0-based) of atoms to retain.
    """
    with open(input_xyz_file, 'r') as infile:
        lines = infile.readlines()

    # The rest lines are atom data
    atom_data = lines[2:]  # Skip the first two lines (atom count and comment)

    # Filter the atom data based on the indices
    filtered_data = [atom_data[i] for i in indices_to_retain if i < len(atom_data)]

    # Update the atom count
    new_num_atoms = len(filtered_data)

    # Write the new XYZ file
    with open(output_xyz_file, 'w') as outfile:
        outfile.write(f"{new_num_atoms}\n")
        outfile.write(lines[1])  # The comment line from the original file
        outfile.writelines(filtered_data)


def optimize_candidate_intermediate(path, input_file, proc):
    """    
    Optimizes the geometry of a candidate intermediate using the XTB method.

    Args:
        path (object): An object containing the parameters required for the optimization (e.g., charge, 
                       multiplicity, solvent, etc.).
        input_file (str): The name of the input XYZ file containing the initial geometry to be optimized.

    Raises:
        RuntimeError: If the XTB optimization process fails.
    """
    os.remove('xtbopt.xyz')
    cmd = f'xtb {input_file} --opt --charge {path.charge} -P {proc} '

    if path.solvent is not None:
        cmd += f'--alpb {path.solvent} '
    if path.multiplicity != 1:
        cmd += f'--uhf {path.multiplicity - 1} '

    try:
        with open(f'candidate_intermediate_opt.out', 'w') as out:
            process = subprocess.Popen(cmd.split(), stderr=subprocess.DEVNULL, stdout=out)
            process.wait()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'Error during XTB optimization: {e}')
        
    os.rename('xtbopt.xyz', f'candidate_intermediate_opt.xyz')
