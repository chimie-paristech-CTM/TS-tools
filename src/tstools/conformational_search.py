import os
import shutil
import argparse
import subprocess
import numpy as np
from tstools.utils import extract_geom_from_xyz, extract_geom_from_crest_ensemble, create_input_file_opt_g16, \
    extract_g16_energy, create_input_file_sp_g16


def get_args():
    """
    Parse command line arguments.

    Returns:

        argparse.Namespace: Parsed command line arguments.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='File to extract geometry')
    parser.add_argument('--opt-DFT', default=False, action="store_true", help='Subsequently optimization of the 10 conformers with the lowest energy, obtained from CREST to determine the most stable conformer')
    parser.add_argument('--atomlist', default=None, nargs="+", help='Constrained atoms')
    parser.add_argument('--charge', default=0, type=int, help='Charge of the molecular system')
    parser.add_argument('--uhf', default=0, type=int, help='Number of unpaired electrons of the molecular system')
    parser.add_argument('--mem', action='store', type=str, default='16GB', help='Specifies the memory requested in the Gaussian16 .com files')
    parser.add_argument('--proc', action='store', type=int, default=8, help='Number of CPU that will be used')    
    parser.add_argument('--functional', action='store', type=str, default='UB3LYP', help='Functional')
    parser.add_argument('--basis-set', action='store', type=str, default='6-31G', help='Basis set')
    parser.add_argument('--solvent', action='store', type=str, default=None, help='Solvent')    

    return parser.parse_args()


def run_crest(name, charge, uhf, atomlist, proc, solvent):
    """
    Run a conformational search using CREST.

    Args:

        name (str): Name of the output file.
        charge (int): Charge of the molecular system.
        uhf (int): Number of unpaired electrons of the molecular system.
        atomlist (list): Constrained atoms
        proc (int): Number of CPU that will be used
        solvent (str): Solvent

    Returns:

        None
    """    

    if solvent:
        kwd_solvent = f"--alpb {solvent}"
    else:
        kwd_solvent = " "

    if atomlist:
        print_constraints_inp(name, atomlist)
        command_line = f"crest {name}.xyz --gfn2 -T {proc} --noreftopo --cinp constraints.inp {kwd_solvent} --chrg {charge} --uhf {uhf} > {name}.out"
    else:
        command_line = f"crest {name}.xyz --gfn2 -T {proc} --noreftopo {kwd_solvent}  --chrg {charge} --uhf {uhf} > {name}.out"

    with open(f'{name}.out', 'w') as out:
        subprocess.run(command_line, shell=True, stdout=out, stderr=subprocess.DEVNULL,)


def print_constraints_inp(name, atomlist):
    """
    Print the constraint file for CREST calculation.
    
    Args:

        name (str): Name of the xyz file
        atomlist (list): Constrained atoms

    Returns:

        None
    """
    constraints = []

    geom = extract_geom_from_xyz(f"{name}.xyz")

    for i in range(0, len(atomlist), 2):
        idx_atm_1 = int(atomlist[i])
        idx_atm_2 = int(atomlist[i + 1])
        _, x1, y1, z1 = geom[idx_atm_1].split()
        _, x2, y2, z2 = geom[idx_atm_2].split()
        coord_atm_1 = np.array([x1, y1, z1], dtype=float)
        coord_atm_2 = np.array([x2, y2, z2], dtype=float)
        distance = np.linalg.norm(coord_atm_1 - coord_atm_2).round(3)
        constraints.append((idx_atm_1 + 1, idx_atm_2 + 1, distance))

    with open('constraints.inp', 'w') as file:
        file.write("$constrain\n")
        file.write("  force constant=0.5\n")
        for constraint in constraints:
            file.write(f"  distance: {constraint[0]}, {constraint[1]}, {constraint[2]}\n")
        file.write("$end")


def run_g16_opt(name, num_conf, charge, uhf, atomlist, mem, proc, functional, basis_set, solvent):
    """
     Run an optimization calculation with G16
 
    Args:

        name (str): Name of the output file.
        num_conf (int): Maximum number of conformers to optimize if dft_validation is set
        charge (int): Charge of the molecular system.
        uhf (int): Number of unpaired electrons of the molecular system.
        atomlist (list): Constrained atoms
        mem (str): Specifies the memory requested in the Gaussian16 .com files
        proc (int): Number of CPU that will be used
        functional (str): functional
        basis_set (str): basis_set
        solvent (str): solvent
 
    Returns:

        None
    """

    if atomlist:
        modredundant = formatting_constraints(atomlist)
    else:
        modredundant = None
    geoms = extract_geom_from_crest_ensemble('crest_conformers.xyz', num_conf)
    multiplicity = int(2 * (uhf * 1/2) + 1)
    energies = []
 
    for idx, geom in enumerate(geoms):
        
        if not geom:
            break    
        tmp_name = f"{name}_conf_{idx}"
        command_line = f"g16 < {tmp_name}.com",
        create_input_file_opt_g16(name=tmp_name, geom=geom, charge=charge, multiplicity=multiplicity, mem=mem, proc=proc, modredundant=modredundant, functional=functional, basis_set=basis_set, solvent=solvent)
        with open(f'{tmp_name}.out', 'w') as out:
            subprocess.run(command_line, shell=True, stdout=out, stderr=subprocess.DEVNULL,)

        energy = extract_g16_energy(f'{tmp_name}.out')
        
        if energy:
            energies.append((idx, energy))
    
    if energies:
        sorted_energies = sorted(energies, key=lambda x: x[1])
        os.mkdir(f'lowest_dft')
        shutil.copy(f'{name}_conf_{sorted_energies[0][0]}.out', 'lowest_dft/dft_best.log')


def run_g16_sp(name, num_conf, charge, uhf, mem, proc, functional, basis_set, solvent):
    """
     Run a single-point calculation with G16

    Args:

        name (str): Name of the output file.
        num_conf (int): Maximum number of conformers to optimize if dft_validation is set
        charge (int): Charge of the molecular system.
        uhf (int): Number of unpaired electrons of the molecular system.
        mem (str): Specifies the memory requested in the Gaussian16 .com files
        proc (int): Number of CPU that will be used
        functional (str): functional
        basis_set (str): basis_set
        solvent (str): solvent

    Returns:

        None
    """

    geoms = extract_geom_from_crest_ensemble('crest_conformers.xyz', num_conf)
    multiplicity = int(2 * (uhf * 1 / 2) + 1)
    energies = []

    for idx, geom in enumerate(geoms):

        if not geom:
            break
        tmp_name = f"{name}_sp_conf_{idx}"
        command_line = f"g16 < {tmp_name}.com",
        create_input_file_sp_g16(name=tmp_name, geom=geom, charge=charge, multiplicity=multiplicity, mem=mem,
                                 proc=proc, functional=functional, basis_set=basis_set, solvent=solvent)
        with open(f'{tmp_name}.out', 'w') as out:
            subprocess.run(command_line, shell=True, stdout=out, stderr=subprocess.DEVNULL, )

        energy = extract_g16_energy(f'{tmp_name}.out')

        if energy:
            energies.append((idx, energy))

    if energies:
        sorted_energies = sorted(energies, key=lambda x: x[1])
        os.mkdir(f'lowest_dft')
        shutil.copy(f'{name}_sp_conf_{sorted_energies[0][0]}.out', 'lowest_dft/dft_best.log')


def formatting_constraints(atomlist):
    """
        Formatting constraint atoms for G16 input.

        Args:

            atomlist (list): Constrained atoms

        Returns:

            None
        """


    constraints = []
    for i in range(0, len(atomlist), 2):
        idx_atm_1 = int(atomlist[i])
        idx_atm_2 = int(atomlist[i + 1])
        constraints.append(f"B  {idx_atm_1}  {idx_atm_2}  F")

    return constraints


def normal_termination_crest(filename):
    """
        Checks if a CREST output file indicates normal termination.

        Args:

            filename (str): Name of the CREST output

        Returns:

            bool: True if normal termination, False otherwise.


        """
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if ' CREST terminated normally.' in line:
            return True

    return False

