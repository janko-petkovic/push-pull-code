'''
Checks if the underlying distribution is log-normal with a
Lilliefors test on the logarythm of the distribution
'''

import numpy as np
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import norm


def check_lognormal(distr):
    log_distr = np.log(distr)

    test_res = lilliefors(log_distr)
    return test_res

