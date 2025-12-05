"""
Key method for constructing the path pool of directories from which the data
will be taken and dataframed subsequently
"""

import os
from typing import Callable, Optional

def create_path_pool(root: str,
                     files_to_include: list[str],
                     additional_checks: Optional[list[Callable]] = []) -> list[str]:
    """
    Given a starting point (root) find all the folders that contain the
    necessary data (files_to_include) while respecting additional constraints
    on their path (additional_checks)

    Parameters
    ----------
    root : str
        Root directory containing all the data

    files_to_include : list of str
        List of srings with the names of the files that must be present
        in the selected directory paths

    additional_checks : list of Callable[[str], bool], optional
        Additional criteria that the path has to respect to be selected

    Returns
    -------
    list of str
        The list of paths to the directories that can be used to build the
        dataset
    """

    #
    # Check that the paths contain all the necessary files (AND logic)
    #
    first_file = files_to_include[0]
    other_files = files_to_include[1:]

    good_paths = []

    # First time I have to use os.walk
    print("-> Looking for the paths with the necessary files...", end="")

    for dir in os.walk(root):
        for sub_file_name in dir[2]:
            if first_file in sub_file_name:
                good_paths.append(dir[0])
                break

    # Now I can just iterate through the paths I already have
    for other_file in other_files:
        temp_paths = []

        for path in good_paths:
            files = os.listdir(path)
            for f in files:
                if other_file in f:
                    temp_paths.append(path)
                    break

        good_paths = temp_paths.copy()

    print(f"done!\n   Retreived {len(good_paths)} candidate paths. ")
    
    #
    # Check names and attitional conditions
    # I should swap the loops but I want to log better
    #
    
    if additional_checks:
        print('\n-> Applying additional path checks...')
        for check in additional_checks:
            print(f"   {check.__name__}:", end="")
            temp_paths = []
            n_dropped = 0
            dropped_str = ""

            for path in good_paths:
                if check(path):
                    temp_paths.append(path)
                else:
                    n_dropped += 1
                    dropped_str += f"\n      {path}"
            
            good_paths = temp_paths

            if n_dropped:
                print(dropped_str)
                print(f"      Dropped {n_dropped} paths.")
            else:
                print(" all good!")
    else:
        print('\n-> No additional path checks to be applied.')

    print(f'\n=> Final number of paths: {len(good_paths)}')

    return good_paths

if __name__ == "__main__":
    _ = create_path_pool(root = "/home/janko/code/protein-distr/KanaanProtData/",
                         files_to_include=["Synapse_l", "ackground"])

                

    
