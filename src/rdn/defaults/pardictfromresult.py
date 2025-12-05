import re
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from pypesto.store import read_from_hdf5

def _df_from_result(path: str,
                    run_index: int):
    '''
        Parse the fitting results into a dataframe (no history for now)

        Parameters
        ----------
        path : str
            Path to .hdf5 result file

        run_index : int
            Index of the run we want to load the parameters of

        Returns
        -------
        pandas DataFrame
            Fitted parameters parsed into a dataframe
        '''
    
    result = read_from_hdf5.read_result(
        filename=path,
        problem=True, optimize=True
    )


    # Parse the specific quantities in a dictionary
    # HERE WE ALSO SWITCH FROM LOGARITHMS TO LINEAR VALUES
    pars = np.array(result.optimize_result.as_list()[run_index].x)
    pars = 10**pars
    parnames = np.array(result.problem.x_names)

    kbs = []
    nbs = []

    pardict = defaultdict(dict)

    for name, par in zip(parnames, pars):

        # Basal
        res = re.search('([0-9]+)_(\w)(\w)o', name)
        if res is not None:
            nss, korn, sorb = res.groups()
            pardict[int(nss)].setdefault(korn+sorb, []).append(par)
            continue

        # Omega
        res = re.search('([0-9]+)_Oo', name)
        if res is not None:
            nss = res.groups()[0]
            pardict[int(nss)]['OoOlast'] = par
            continue

        # Pi
        res = re.search('([0-9]+)_Pi', name)
        if res is not None:
            nss = res.groups()[0]
            pardict[int(nss)]['Pi'] = par

        
    # Create the dataframe, make arrays out of lists
    df = pd.DataFrame(pardict).T
    df.loc[:, 'Kb'] = df['Kb'].map(lambda x: np.array(x))
    df.loc[:, 'Nb'] = df['Nb'].map(lambda x: np.array(x))

    # Add the global quantities to the dataframe
    df.loc[:,'KsoONbl'] = pars[np.where(parnames == 'KsoONbl')].item()
    df.loc[:,'NsoNbl'] = pars[np.where(parnames == 'NsoNbl')].item()
    df.loc[:, 'tau_K'] = pars[np.where(parnames=='tau_K')].item()
    df.loc[:,'sigma_K'] = pars[np.where(parnames=='sigma_K')].item()
    df.loc[:,'tau_N'] = pars[np.where(parnames=='tau_N')].item()
    df.loc[:,'sigma_N'] = pars[np.where(parnames=='sigma_N')].item()

    return df

def _get_basal_lognormal_distribution(Xbs, ax, **kwargs):
    '''
    Assuming that Xb is sampled from a lognormal distribution,
    return the mean and std of the underlying normal
    distribution.

    Parameters
    ----------
    Xb : numpy array
        Basal sampling

    ax : plt axis
        If not None, plot the log(Xb) distribution with the
        corresponding normal fit

    Returns
    -------
    tuple 
        (mu_log_X, std_log_X)
    '''

    # Get the logs
    log_Xbs = np.log(Xbs)

    print(f'- Sample size: {len(log_Xbs)}')

    # Test for normality of logs
    # norm_res = stats.shapiro(log_Xbs)
    norm_res = stats.normaltest(log_Xbs)
    anderson_res = stats.anderson(log_Xbs)
    dist = stats.norm
    res = dist.fit(log_Xbs)

    # We follow wikipedia for a sanity check
    mu_log_X = res[0]
    std_log_X = res[1]

    mean_X = np.exp(mu_log_X + 0.5*std_log_X**2)
    std_X = ((np.exp(std_log_X**2) - 1) * (np.exp(2*mu_log_X + std_log_X**2)))**0.5

    print(f'- Mean: {mean_X} +- {std_X}')
    print(f'- Anderson test on log: {anderson_res}')

    # Plotting 
    if ax is not None:
        c = 'tab:orange' if norm_res[1] >= 0.05 else 'gray'
        _, bins, _ = ax.hist(log_Xbs, bins=10, density=True, color='tab:blue')
        ax.plot(bins, dist.pdf(bins, *res), c=c, linewidth=3)
        ax.set_xlabel(kwargs['x_label'])
        ax.set_ylabel('pdf')


    return mu_log_X, std_log_X

def pardict_from_result(path: str, 
                        Chi: float, 
                        dendrite_length: float, 
                        N_mean: float,
                        run_index: int,
                        plot_basal_distributions: bool = False,
                        ):
    '''
    Build a simulation parameter dictionary from a hdf5 result.

    Parameters
    ----------
    path : str
        Path to .hdf5 result file
    chi : float
        Assumed ratio k_K/k_N
    dendrite_length : float
        Assumed dendritic length
    N_mean : float
        Assumed experimental average for N (calcineurin Helm et al.)
    run_index : int
        Index of the run we want to load the parameters of
    plot_basal_distributions: bool
        If True, generate a plot to show the distributions of the inferred
        basal catalytic values


    Returns
    -------
    dict
        Parameter dictionary in the right format for the simulations

    Note
    ----
    This method looks like a good place from which to import the history
    too and do the best pardict selection.
    '''

    # Build the parameter dataframe, apply assumptions
    df = _df_from_result(path, run_index)
    Omega = dendrite_length/Chi

    # Build the pardict: start with global parameters
    pardict = {}
    pardict['Chi'] = Chi
    pardict['Ks'] = df['KsoONbl'].iloc[0].item()*Omega*N_mean
    pardict['Ns'] = df['NsoNbl'].iloc[0].item()*N_mean
    pardict['tau_K'] = df['tau_K'].iloc[0].item()
    pardict['sigma_K'] = df['sigma_K'].iloc[0].item()
    pardict['tau_N'] = df['tau_N'].iloc[0].item()
    pardict['sigma_N'] = df['sigma_N'].iloc[0].item()
    pardict['Pi'] = df['Pi'].mean()


    # Basal distributions: recover the underlying lognorm parameters,
    # and also check for lognormality and plot what is happening

    if plot_basal_distributions:
        fig, axs = plt.subplots(1,2, figsize=(8,3.5), dpi=300)
        fig.tight_layout(pad=3)
    else:
        axs = [None, None] # dirty but works

    # Kb distribution
    print('\nBasal K distribution:')
    Kbs = np.concatenate(df['Kb'].to_list())*Omega*N_mean

    mu_log_K, std_log_K = _get_basal_lognormal_distribution(Kbs, axs[0], x_label=r'log $K_b$')

    # Extract Nb distribution
    print('\nBasal N distribution:')
    Nbs = np.concatenate(df['Nb'].to_list())*N_mean

    mu_log_N, std_log_N = _get_basal_lognormal_distribution(Nbs, axs[1], x_label=r'log $N_b$')
    
    # Compute the covariance between log_Kb and log_Nb
    r_matrix = np.corrcoef(np.log(Kbs[:-1]), np.log(Nbs))
    cov_log_K_N = np.cov(np.log(Kbs[:-1]), np.log(Nbs))
    
    print('\nCorrelation matrix between log K_b and log N_b')
    print(r_matrix)

    pardict['mu_log_K_N'] = np.array([mu_log_K, mu_log_N])
    pardict['cov_log_K_N'] = cov_log_K_N
    
    return pardict
