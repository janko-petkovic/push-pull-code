from os.path import join
import torch
torch.manual_seed(2023)

import numpy as np
np.random.seed(2023)

# from math import sqrt
import matplotlib.pyplot as plt
import scipy.stats as stats

from .. import Simulation




def GodaHoVsHe(model, model_p_dict, save_path=None):

    ################
    # Control panel
    ################

    simulation_time = 40
    spine_number = 200
    inter_spine_distance = 1

    stim_configurations = [
        torch.arange(10,11,4),
        torch.arange(10,19,4),
        torch.arange(10,27,4),
        torch.arange(10,37,4),
        torch.arange(10,67,4),
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

    r = 5
    t_indexes = [2,10,20,30,40]

    # Norm
    avg_homo_rel_batches = []
    avg_hetero_rel_batches = []

    for rel_batch, stim_config in zip(exp_rel_size_batches, stim_configurations):

        avg_homo_rel_batch = rel_batch[:, stim_config, :].mean(axis=1)
        avg_hetero_rel_batch = rel_batch[:, stim_config[-1]:stim_config[-1]+r, :].mean(axis=1)

        avg_homo_rel_batches.append(avg_homo_rel_batch)
        avg_hetero_rel_batches.append(avg_hetero_rel_batch)


    ###########
    # Plotting and analysis
    ###########


    fig, axs = plt.subplots(1,5,figsize=(20,4), dpi=100)
    fig.tight_layout(pad=5, rect=[0.1,0.1,0.9,0.95])
    fig.suptitle(r'Fig. 1E (40 min), Oh et al. 2015')

    for ax, t_index in zip(axs, t_indexes):
        for horb, herb, stim_config in zip(avg_homo_rel_batches, avg_hetero_rel_batches, stim_configurations):
            homos = horb[t_index]
            heteros = herb[t_index]

            s,i,r,p,e = stats.linregress(homos, heteros)

            points = ax.scatter(homos, heteros, marker='.')

            x = torch.linspace(homos.min(), homos.max(), 10)
            y = s*x + i

            if p<0.05:
                c = points.get_facecolor()
                ax.plot(x,y, label=f'{len(stim_config)} stim, r = {r:.2}',c=c)
                ax.legend()

            else:
                ax.plot(x,y,c='gray', alpha=0.3)
                points.set_facecolor('white')
                points.set_edgecolor('gray')
                points.set_alpha(0.3)

            ax.axhline(y=1, linestyle='--', c='black', linewidth=0.5)
            ax.axvline(x=1, linestyle='--', c='black', linewidth=0.5)

            ax.set_title(f'{t_index+2} min')
            ax.set_xlabel('Stim spine GFP\n(f baseline)')
            ax.set_ylabel('Hetero spine GFP\n(f baseline)')

    if save_path:
        plt.savefig(join(save_path, 'goda_ho_vs_he.png'))

    plt.show()
