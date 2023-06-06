import os
from typing import Callable
from functools import wraps


def work_in(dir_ext: list) -> Callable:
    """
    Decorator to execute a function in a different directory.

    Args:
        dir_ext (list: List containing subdirectory name to create or use.

    Returns:
        Callable: Decorated function.
    """

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):

            here = os.getcwd()
            dir_path = os.path.join(here, dir_ext[0])

            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)

            os.chdir(dir_path)
            try:
                result = func(*args, **kwargs)
            finally:
                os.chdir(here)

                if len(os.listdir(dir_path)) == 0:
                    os.rmdir(dir_path)

            return result

        return wrapped_function

    return func_decorator

def xyz_to_gaussian_input(xyz_file, output_file, method='UB3LYP', basis_set='6-31G(d,p)', extra_commands='opt=(calcfc,ts, noeigen) freq=noraman', IRC=True):
    """
    Convert an XYZ file to Gaussian 16 input file format.

    Args:
        xyz_file (str): Path to the XYZ file.
        output_file (str): Path to the output Gaussian input file to be created.
        method (str, optional): The method to be used in the Gaussian calculation. Default is 'B3LYP'.
        basis_set (str, optional): The basis set to be used in the Gaussian calculation. Default is '6-31G(d)'.
    """
    filename = xyz_file.split('/')[-1].split('.xyz')[0]

    with open(xyz_file, 'r') as xyz:
        atom_lines = xyz.readlines()[2:]  # Skip the first two lines (number of atoms and comment)

    with open(output_file, 'w') as gaussian_input:
        # Write the route section
        gaussian_input.write(f'%Chk={filename}.chk\n%nproc=8\n%Mem=16GB\n# {method}/{basis_set} {extra_commands}')
        
        # Write the title section
        gaussian_input.write('\n\nTitle\n\n')

        # Write the charge and multiplicity section
        gaussian_input.write('0 1\n')

        # Write the Cartesian coordinates section
        for line in atom_lines:
            atom_info = line.split()
            element = atom_info[0]
            x, y, z = atom_info[1:4]
            gaussian_input.write(f'{element} {x} {y} {z}\n')

        # Write the blank line and the end of the input file
        gaussian_input.write('\n')

        if IRC:
            gaussian_input.write('--Link1--\n')
            gaussian_input.write(f'%Chk={filename}.chk\n%nproc=8\n%Mem=16GB\n# {method}/{basis_set} IRC(MaxPoints=20,CalcAll,Synchronous)')
            gaussian_input.write('\n\nTitle\n\n')
            gaussian_input.write('0 1\n\n')

    print(f'Gaussian input file "{output_file}" has been created.')