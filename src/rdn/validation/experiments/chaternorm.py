from os.path import join
import torch
torch.manual_seed(2023)

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .. import Simulation




def chater_norm(model,
                model_p_dict,
                simulation_time: int = 40,
                spine_number: int = 100,
                inter_spine_distance: int = 1,
                axs: list = [],
                save_folder: str = ''):

    ################
    # Control panel
    ################

    simulation_time = 40
    spine_number = 1000
    inter_spine_distance = 1

    stim_configurations = [
        torch.arange(10,11,5),
        torch.arange(10,19,4),
        torch.arange(10,60,10),
        torch.arange(10,45,5),
        torch.arange(10,115,7),
    ]


    ################
    # Run experiment
    ################

    exp_basal_size_batches = []
    exp_size_batches = []
    exp_rel_size_batches = []
    
    for stim_idxes in stim_configurations:
        simulation = Simulation(model = model,
                                model_p_dict = model_p_dict,
                                simulation_time = simulation_time,
                                spine_number = spine_number,
                                inter_spine_distance = inter_spine_distance,
                                stim_indexes = stim_idxes)

        basal_sizes_batch, sizes_batch, rel_sizes_batch = simulation.run(10)

        exp_basal_size_batches.append(basal_sizes_batch)
        exp_size_batches.append(sizes_batch)
        exp_rel_size_batches.append(rel_sizes_batch)


    ################
    # Gather the data
    ################
    
    nsss = [len(s) for s in stim_configurations]
    t_idx = 2
    r = 10


    # BASELINE: RID
    tot_base_RID_per_nss = []
    close_base_RID_per_nss = []
    stim_base_RID_per_nss = []

    for batch, stim_config in zip(exp_basal_size_batches, stim_configurations):
        tot_base_RID_per_nss.append(batch[t_idx])
        stim_base_RID_per_nss.append(batch[t_idx][stim_config])

        close_base_RID_per_nss.append(batch[t_idx][stim_config[0]-r:stim_config[-1]+r:1])

    # Norm
    close_norm_per_nss = []
    tot_norm_per_nss = []
    stim_norm_per_nss = []

    for batch, stim_config in zip(exp_rel_size_batches, stim_configurations):
        tot_norm_per_nss.append(batch[t_idx])
        stim_norm_per_nss.append(batch[t_idx][stim_config])
        close_norm_per_nss.append(batch[t_idx][0:stim_config.max()+r])


    ###########
    # Plotting
    ###########

    # Format the simulation data
    def prepare_plotting_statistic(data):
        means = np.array([d.mean().numpy() for d in data])
        errs = np.array([d.std().numpy()/sqrt(len(d.flatten())) for d in data])
        nsss = np.array([len(sc) for sc in stim_configurations])

        return np.stack([nsss, means, errs], axis=0)

    sim_tot = prepare_plotting_statistic(close_norm_per_nss)
    sim_stim = prepare_plotting_statistic(stim_norm_per_nss)
    data_tot = np.load('data/validation_data/goda_data/tot_norm_per_nss.npy')
    data_stim = np.load('data/validation_data/goda_data/stim_norm_per_nss.npy')

    datas = [sim_tot,
             sim_stim,
             data_tot,
             data_stim]


    # Plotting auxiliary function
    rdngreen = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    def fit_and_plot(data, ax, **kwargs):

        # Linear fit This is stupid but whatever
        def lin_func(x, a, b):
            return a + b*x

        def residue(p, X, Y, Y_err):
            return (p[0] + p[1]*X - Y)/Y_err

        # res = least_squares(residue, [1.,0.], args=(data[0], data[1], data[2]))
        popt, pcov = curve_fit(lin_func, data[0], data[1], p0=[0.,1.], sigma=data[2])
        intercept, slope = popt
        err_in, err_sl = np.sqrt(np.diag(pcov))
       
        # see if there is a trend there
        if (abs(slope) - 1.5*abs(err_sl) < 0):
            linestyle = '--'
            label_line = 'Linear fit (slope n.s.)'
        else:
            linestyle = '-'
            label_line = 'Linear fit (slope *)'

        # Plot
        x_line = np.linspace(0, data[0].max()*1.1)
        ax.errorbar(data[0], data[1], data[2], fmt='.',
                    markersize=9, elinewidth=1, mew=2,
                    c=kwargs.get('c'), label=kwargs.get('label'))
        
        ax.plot(
            x_line, intercept + slope*x_line,
            c=kwargs.get('c'), linestyle=linestyle, linewidth=2,
            label=label_line)

        ax.fill_between(x_line,
                        (intercept-err_in) + (slope - err_sl)*x_line,
                        (intercept+err_in) + (slope + err_sl)*x_line,
                        color=kwargs.get('c'), alpha=0.03
                        )

        ax.set_xlim(0, data[0, -1]+1)
        ax.set_ylim(kwargs.get('y_lim'))
        ax.set_xticks(data[0])
        ax.set_title(kwargs.get('title'))
        ax.set_ylabel(kwargs.get('y_label'))
        ax.legend()


    if len(axs) == 0:
        fig, axs = plt.subplots(2,2,figsize=(8,6), dpi=100)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.98,
                            hspace=0.8, wspace=0.5)
    
    labels = ('Simulated data',
              'Simulated data',
              'Experimental data',
              'Experimental data')

    y_labels = (
            r'Total $P/P_{pre}$',
            r'Stim $P/P_{pre}$',
            r'Total $RID/RID_{pre}$',
            r'Stim $RID/RID_{pre}$')

    for idx, (data, ax, label, y_label) in \
            enumerate(zip(datas, axs.T.flatten(), labels, y_labels)):
        c = 'black' if idx>1 else rdngreen
        y_lim = (1.2,2.2) if (idx%2) else (1,1.45)

        fit_and_plot(data, ax, c=c, y_lim=y_lim, label=label, y_label=y_label)

    for ax in axs.flatten():
        ax.set_xlabel('Number of stimulations')

    axs[0,0].legend(loc='upper left')
    axs[0,1].legend(loc='upper left')

    if save_folder:
        plt.savefig(join(save_folder, 'sim_chater_norm.png'))

    plt.show()
