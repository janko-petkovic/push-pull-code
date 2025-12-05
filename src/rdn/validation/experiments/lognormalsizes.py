from os.path import join
import matplotlib.pyplot as plt
import torch
torch.manual_seed(2023)
import numpy as np

import scipy.stats as stats

from .. import Simulation


def lognormal_sizes(model, 
                      model_p_dict,
                      simulation_time: int = 40,
                      spine_number: int = 100,
                      position_stim: int = 50,
                      position_unstim: int = 54,
                      inter_spine_distance: int = 1,
                      axs: list[plt.Axes] = [],
                      save_folder: str = ''):

    ################
    # Control panel
    ################
    stim_configurations = [
        torch.tensor([50]),
    ]

    ################
    # Run experiment
    ################

    sizes = []

    basal_size_batch = []
    size_batch = []
    rel_size_batch = []
    
    for stim_idxes in stim_configurations:
        simulation = Simulation(model = model,
                                model_p_dict = model_p_dict,
                                simulation_time = simulation_time,
                                spine_number = spine_number,
                                inter_spine_distance = inter_spine_distance,
                                stim_indexes = stim_idxes)

        # simulation.visualize_run(100,2)

        basal_sizes, sizes, rel_sizes = simulation.run(100)

        basal_size_batch.append(basal_sizes.squeeze())
        size_batch.append(sizes.squeeze())
        rel_size_batch.append(rel_sizes.squeeze())

    rsb = rel_size_batch[0]
    sem_rs, avg_rs = torch.std_mean(rsb, axis=2)

    bs = basal_size_batch[0][2].flatten()
    log_bs = torch.log(bs)
    
    norm_test_res = stats.normaltest(log_bs)
    norm_fit_res = stats.norm.fit(log_bs)
    

    if len(axs)==0:
        fig, ax = plt.subplots(1,1, figsize=(4,3), dpi=100)
        ax2 = fig.add_axes([0.65, 0.45, 0.3,0.3])

        fig.subplots_adjust(left=0.13, bottom=0.2, right=0.98, top=0.9)
        axs = [ax, ax2]

    ax = axs[0]
    ax.hist(bs, density=True, label='Predicted P', bins=50)
    ax.set_xlabel(r'$P$')
    ax.set_ylabel(r'pdf')


    ax = axs[1]
    _, bins, _ = ax.hist(log_bs, density=True, label='Predicted P')

    x_plot = np.linspace(bins.min()*0.9, bins.max()*1.1, 100)
    c = 'gray' if norm_test_res.pvalue < 0.05 else 'tab:orange'
    ax.plot(x_plot, stats.norm.pdf(x_plot, *norm_fit_res), c=c,
            linewidth=3,
            label=f"D'agostino p-value = {norm_test_res.pvalue:.2}")

    ax.legend(loc=[-0.6,1.2])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r'$\log P$')


    if save_folder:
        plt.savefig(join(save_folder, 'lognormal_sizes.png'))
    
    if not save_folder:
        plt.savefig('lognormal_sizes.png')
    
    plt.show()
