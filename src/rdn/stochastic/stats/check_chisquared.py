'''
Checks if the underlying distribution is chi squared with a
Lilliefors test on randomly selected positive/negative
square roots of the given distr values

https://stats.stackexchange.com/questions/125648/transformation-chi-squared-to-normal-distribution
'''

import numpy as np
from statsmodels.stats.diagnostic import lilliefors

import matplotlib.pyplot as plt


def check_chisquared(distr):
    n_spines = distr.shape

    # Square roots
    root_distr = np.sqrt(distr)

    # Square roots can be positive or negative
    sign_distr = np.random.binomial(1,0.5,n_spines)*2 - 1
    pm_root_distr = root_distr * sign_distr

    test_res = lilliefors(pm_root_distr)
    return test_res

