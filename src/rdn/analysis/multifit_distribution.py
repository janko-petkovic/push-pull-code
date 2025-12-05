import numpy as np
import scipy.stats as scs


def multifit_distribution(distr: np.ndarray) -> dict:
    '''
    Fit the given distribution with a number of test functions.
    Actually good, you can give him any data and he will fit the stuff.
    '''

    fit_fns = {
        'norm' : scs.norm,
        'expon' : scs.expon,
        'maxwell' : scs.maxwell,
        'gamma' : scs.gamma,
        'lognorm' : scs.lognorm,
        'weibull_min': scs.weibull_min,
        'chi2': scs.chi2,
        'chi': scs.chi,
        'burr12' : scs.burr12,
        'gausshyper' : scs.gausshyper,
        'fisk' : scs.fisk,
        'logistic' : scs.logistic,
        }

    fit_params = {}

    for name, fn in fit_fns.items():
        params = fn.fit(distr, method = "MLE", floc=0)
        fit_params[name] = params

    return fit_params




