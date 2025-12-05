"""
A custom implementation of the multinomial generator.
I did it because np.random.multinomial does not support np.ndarrays
as input for n.

Validation with np.random.multinomial:
    n = np.ones(1e5)*1e5
    pvals = [0.1]*10

    Compared the distribution of KS p-values obtained with
    1. 2 sets of samples generated with numpy
    2. samples generated with numpy and multi_multinomial
    * Ks (statistic = 0.0084, pvalue=0,872)

    
Quick time profiling against a for loop with np.random.multinomial():
    custom method scales better with the length of n
    n = 2: time = 1.25 numpy time
    n = 100: time = 0.25 numpy time
    n = 5000: time = 0.06 numpy time
"""

import numpy as np


def _multi_multinomial(n: np.ndarray,
                       pvals: np.ndarray) -> np.ndarray:
    
    # Calculate the binomial for the first event to happen
    results = []
    first_event_counts = np.random.binomial(n, pvals[:,0])
    n_left = n-first_event_counts
    results.append(first_event_counts)

    # If the complementary event has more than one event in it than recurr
    # the function using the remaining experiments and (normalized) pvals
    if len(pvals[0,1:]) > 1: 
        results += _multi_multinomial(n_left,
                                      pvals[:,1:]/pvals[:,1:].sum(axis=1).reshape(-1,1))

    else:
        results.append(n_left)

    return results


def multi_multinomial(n: list[int],
                      pvals: np.ndarray ) -> np.ndarray:
    """
    Parameters
    ----------
    n : array_like of int, length N
        Array containing the numbers of different experiments

    pvals : (N, p) numpy array
        Probabilities of each of the p events to occur for every value
        of n.
        The values of the given prbabilities have to add up to 1 for each n.
        If their sum is smaller, the missing mass will be added to the
        last pval.
        If their sum is bigger an error is thrown.

    Returns
    -------
    (N, p) numpy array 
        An array of shape (N,p) with each row containing the the counts 
        of the drawn samples for the corresponding n. 
    """

    # First cast everything into a numpy array
    n = np.asarray(n)

    # If pvals is not a matrix
    if len(pvals.shape) != 2:
        raise ValueError("pvals has to be a matrix")

    # Check if it is compatible with n
    if (len(pvals) != len(n)):
        raise ValueError("Provided n and pvals are not compatible")

    # Check if any of the rows sum to more than 1
    pvals_sum = pvals.sum(axis=1)
    if np.where(pvals_sum>1,1,0).any():
        bad_idxes = np.where(pvals_sum>1)[0]
        raise ValueError(f"Following pvalue rows add up to more than 1: {bad_idxes}")

    # If rows do not add up to one, add the residue to the last
    # pval in the row
    residuals = 1 - pvals_sum
    pvals[:,-1] += residuals

    # Check if the previous operation has gone wrong due to rounding errors
    pvals_sum = pvals.sum(axis=1)
    fails = np.logical_not(np.isclose(pvals_sum, 1))
    if fails.any():
        fail_idxes = np.where(fails)[0]
        raise ValueError(f"Following rows failed to add up to 1: {fail_idxes}")
   
    return np.stack(_multi_multinomial(n, pvals), axis=0).T




if __name__ == '__main__':
    n = np.ones(10, dtype=int)*10
    p = np.ones((10,5))*0.2
    print(multi_multinomial(n,p))
