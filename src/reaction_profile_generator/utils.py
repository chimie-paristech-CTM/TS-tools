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
