"""
This file contains the generators for the functions to be used as additional
conditions in the create_path_pool method
"""

from typing import Callable

import scipy.stats as scs

from .aux import eggl_import, eggl_transform



def generate_check_t_series_corr(col_1: str,
                                 col_2: str,
                                 corr_type: str,
                                 min_r: float,
                                 max_p: float) -> Callable[[str],bool]:
    """
    Generate a function that checks if the t_series located in two different
    columns of the dataframe generated from a path argument are correlated 
    for each row. We use this cos our data has t_series for luminosities and
    we transform these series (background subtraction, normalization, dunno)
    so we want to make sure that the transformations are good.

    Parameters
    ----------
    col_1 : str
        Name of the first column

    col_2 : str
        Name of the second column

    corr_type : str ('pearson' or 'spearmann')
        Correlation test to be used

    min_r : float
        Minimum r value to consider the test passed

    max_p : float
        Maximum p value to consider the test passed

    Returns
    -------
    Callable[[str], bool]
        The function that tests the path passed as argument and returns
        True if passed
    """

    def check_t_series_corr(path: str) -> bool:

        # Build the spine dataframe (each row is a spine)
        df, bg = eggl_import(path)
        df['egglRID'] = eggl_transform(df, bg)

        # Pick the correlation type
        corr_f = scs.pearsonr if corr_type == 'pearson' else scs.spearmanr
       
        # Check the correlations row by row
        for _, row in df.iterrows():
            r, p = corr_f(row[col_1], row[col_2])
            if (r < min_r) or (p > max_p): 
                return False

        return True

    return check_t_series_corr
    


def generate_check_names_in_path(names: str) -> Callable[[str],bool]:

    def check_names_in_path(path: str) -> bool:
        for name in names:
            if name in path: return True

        return False

    return check_names_in_path

def generate_check_names_not_in_path(names: str) -> Callable[[str],bool]:

    def check_names_not_in_path(path: str) -> bool:
        for name in names:
            if name in path: return False

        return True

    return check_names_not_in_path
