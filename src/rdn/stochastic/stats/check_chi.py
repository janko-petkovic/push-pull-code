'''
Checks if the underlying distribution is chi squared with a
Lilliefors test on the square roots of the given distr values
'''

import numpy as np
from statsmodels.stats.diagnostic import lilliefors

import matplotlib.pyplot as plt


def check_chisquared(distr):
    n_spines = distr.shape
    root_distr = np.sqrt(distr)

    test_res = lilliefors(root_distr)
    return test_res

