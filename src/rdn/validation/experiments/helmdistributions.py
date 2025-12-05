from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from ...defaults.pardictfromresult import _df_from_result

def _normtest_and_plot(data, ax, **kwargs):

    # Test for normality (we are passing logarithms)
    # norm_test_res = stats.shapiro(data)
    norm_test_res = stats.normaltest(data)
    norm_fit_res = stats.norm.fit(data)
    
    vals, bins, patches = ax.hist(data, bins=kwargs.get('bins'), density=True,
                         label=kwargs['label'])

    if kwargs.get('hist_color'):
        for rect in patches:
            rect.set_facecolor(kwargs.get('hist_color'))
    
    x_plot = np.linspace(bins.min()*0.9, bins.max()*1.1, 100)
    c = 'gray' if norm_test_res.pvalue < 0.05 else 'tab:orange'
    ax.plot(x_plot, stats.norm.pdf(x_plot, *norm_fit_res), c=c,
            linewidth=3,
            label=f"D'agostino p-value = {norm_test_res.pvalue:.2}")


    ax.set_ylim(0, vals.max()*1.5)
    ax.set_xlabel(kwargs['x_label'])
    ax.set_ylabel('pdf')
    ax.legend(loc='upper left')



def helm_distributions(path_to_pars: str,
                      run_index: int = 440,
                      Omega: float = 10,
                      N_mean: float = 5000,
                      axs: list = [],
                      save_folder: str = '',
                      alt_helm_data_folder: str = '',
                      bins = None) -> None:
    '''
    Parameters
    ----------
    path : str
        Path to .hdf5 result file
    Omega : float
        Omega value obtained from fits
    N_mean : float
        Average Nb value
    run_index : int
        Index of the run we want to load the parameters of
    '''

    df = _df_from_result(path_to_pars, run_index)
    Kbs = np.concatenate(df['Kb'].to_list())*Omega*N_mean
    Nbs = np.concatenate(df['Nb'].to_list())*N_mean
    
    log_Kbs = np.log(Kbs)
    log_Nbs = np.log(Nbs)

    if alt_helm_data_folder:
        log_Cs = np.load(alt_helm_data_folder)
    else:
        log_Cs = np.load('data/validation_data/helm_data/log_camkii.npy')
    datas = (log_Cs, log_Kbs, log_Nbs)

    # Start plotting
    if len(axs)==0:
        fig, axs = plt.subplots(3,1,figsize=(4,9), dpi=100)
        fig.subplots_adjust(left=0.18, bottom=0.07, right=0.98, top=0.98,
                            hspace=0.5)

    labels = ('Experimental data (Helm et al. 2022)', 'Optimized values',
              'Optimized values')
    x_labels = ('Log CaMKII luminosity [a.u.]', r'Log $K_b$', r'Log $N_b$')

    if not bins: bins=10

    for data, ax, label, x_label in zip(datas, axs, labels, x_labels):
        if 'CaMKII' in x_label:
            _normtest_and_plot(data, ax, hist_color='black', bins=bins,
                              label=label, x_label=x_label)
        else: 
            _normtest_and_plot(data, ax, label=label, bins=bins,
                              x_label=x_label)

    if save_folder:
        plt.savefig(join(save_folder, 'helm_distributions.png'))

    plt.show()
     
