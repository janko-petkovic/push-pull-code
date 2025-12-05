from os.path import join
from scipy.optimize import curve_fit
import torch

# from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


from .. import Simulation




def GodaStimDepression(model,
                       model_p_dict,
                       simulation_time: int = 40,
                       spine_number: int = 1210,
                       inter_spine_distance: int = 1,
                       axs = [],
                       save_folder: str = ''):
    ################
    # Control panel
    ################

    stim_configurations = [
        torch.arange(10,11,5),
        torch.arange(10,19,4),
        torch.arange(10,60,10),
        torch.arange(10,45,5),
        torch.arange(0,115,7),
    ]



    ################
    # Run experiment
    ################

    # exp means experiment not exponential
    exp_basal_size_batches = []
    exp_size_batches = []
    exp_rel_size_batches = []
    
    for stim_idxes in stim_configurations:
        simulation = Simulation(model = model,
                                model_p_dict = model_p_dict,
                                simulation_time = simulation_time,
                                spine_number = spine_number,
                                inter_spine_distance = inter_spine_distance,
                                stim_indexes = stim_idxes,
                                p_fail_to_uncage=0.0)

        # simulation.visualize_run(2,2)
        basal_sizes_batch, sizes_batch, rel_sizes_batch = simulation.run(200)

        exp_basal_size_batches.append(basal_sizes_batch)
        exp_size_batches.append(sizes_batch)
        exp_rel_size_batches.append(rel_sizes_batch)
    

    ################
    # Gather the data
    ################
    
    nsss = [len(s) for s in stim_configurations]
    t_idx = 2

    X, Y = [], []

    for bsb, rsb, stim_config in zip(exp_basal_size_batches, exp_rel_size_batches,
                                    stim_configurations):
        X.append(bsb[t_idx, stim_config, :].flatten().tolist())
        Y.append(rsb[t_idx, stim_config, :].flatten().tolist())

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    experiment = np.load('data/validation_data/goda_data/stim_norm_2_vs_base.npy')

    exp_idxes = np.where(X <= experiment[0].max())
    X, Y = X[exp_idxes], Y[exp_idxes]
    sim = np.stack([X, Y], axis=0)
    



    ################
    # Plotting
    ################

    def fit_and_plot(X, Y, axs, **kwargs):
        c = kwargs.get('c')
        label = kwargs.get('label')
        thr = 1.2

        # Linear regression of first plot
        def line(x, i, s):
            return i + x*s

        (i, s), linpcov = curve_fit(line, X, Y, (0,-1))
        errin, errsl = np.sqrt(np.diag(linpcov))
        print(abs(s)-1.95*abs(errsl))

        # Data for second plot
        bins = np.linspace(X.min(), X.max(), 10)
        idxess = np.digitize(X, bins)

        plot_bins = []
        ratios = []

        for idx_value in set(idxess):
            y = Y[np.where(idxess==idx_value)]

            n_nr = y[y<thr].shape[0]
            n_tot = y.shape[0]
            ratio = n_nr / n_tot

            ratios.append(ratio)
            plot_bins.append(X[np.where(idxess==idx_value)].mean())

        # Logistic regression for second plot
        def logistic(x, mu, sigma):
            return 1/(1+np.exp(-(x - mu)/sigma))

        def dlogdmu(x, mu, sigma):
            return logistic(x,mu,sigma) * np.exp(-(x-mu)/sigma)/sigma

        def dlogdsigma(x, mu, sigma):
            return logistic(x,mu,sigma) * np.exp(-(x-mu)/sigma) * (x-mu)/sigma**2

        def err_logisic(x, mu, sigma, errm, errsig):
            return np.sqrt((dlogdmu(x,mu,sigma)*errm)**2 +
                           (dlogdsigma(x,mu,sigma)*errsig)**2)



        (mu, sigma), logpcov = curve_fit(logistic, 
                                         plot_bins, 
                                         ratios, 
                                         p0=[2e4,1e2])

        errm, errsig = np.sqrt(np.diag(logpcov))
        

        ax = axs[0]
        ax.scatter(X,Y, s=5, alpha=0.3, c=c, marker='.', label=label)
        ax.plot(X, i+s*X, c=c, label='Linear fit', linestyle=(0,(5,5)))
        ax.fill_between(np.sort(X),
                       (i-errin) + (s - errsl) * np.sort(X),
                       (i+errin) + (s + errsl) * np.sort(X),
                       color=c, alpha=0.1)
        ax.axhline(y=thr, linestyle=(0,(3,3)), linewidth=1, c=c)

        ax.set_ylim(-0.2, 5)
        ax.legend()

        ax=axs[1]
        fine_plot_bins = np.linspace(min(plot_bins), max(plot_bins)*1.1, 100)
        ax.scatter(plot_bins, ratios, c=c, marker='.')
        ax.plot(fine_plot_bins, logistic(fine_plot_bins, mu, sigma), c=c,
                label='Logistic fit')
        ax.axhline(y=1, linestyle=(0,(3,3)), linewidth=1, c=c)

        low_ic = logistic(fine_plot_bins, mu, sigma) - \
                    err_logisic(fine_plot_bins, mu, sigma, errm, errsig)
        high_ic = logistic(fine_plot_bins, mu, sigma) + \
                    err_logisic(fine_plot_bins, mu, sigma, errm, errsig)
        ax.fill_between(fine_plot_bins, low_ic, high_ic, 
                        color=c, alpha=0.1)

        ax.set_ylim(-0.1,1.1)
        ax.legend(loc='lower right')



    if len(axs) == 0:
        fig, axs = plt.subplots(2,2,figsize=(8,3), height_ratios=(2,1),
                                sharex=True, dpi=100)
        fig.subplots_adjust(left=0.1, bottom=0.2, right=0.98, top=0.98,
                            wspace=0.3)
    rdngreen = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

    cs = (rdngreen, 'black')
    labels = ('Simulated data', 'Experimental data')

    for data, col, c, label in zip([sim, experiment], axs.T, cs, labels):
        fit_and_plot(*data, col, c=c, label=label)

    axs[0,0].set_ylabel(r'$\Delta_{P^{(i)}} \, / \,P^{(i)}_{pre}$')
    axs[1,0].set_xlabel(r'$P^{(i)}_{pre}$ [a.u.]')
    axs[1,0].set_ylabel(r'NR-ratio')
    axs[0,1].set_ylabel(r'$\Delta_{RID^{(i)}} \, / \, RID^{(i)}_{pre}$')
    axs[1,1].set_ylabel(r'NR-ratio')
    axs[1,1].set_xlabel('$RID^{(i)}_{pre}$ [a.u.]')
   

    if save_folder:
        plt.savefig(join(save_folder,'sim_stim_norm_2_vs_base.png'))

    plt.show()
