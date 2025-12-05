from os.path import join
import torch
torch.manual_seed(2023)

import matplotlib.pyplot as plt
import numpy as np

from .. import Simulation




def scanziani_mc(model,
                 model_p_dict_orig,
                 simulation_time: int = 40,
                 spine_number: int = 100,
                 position_stim: int = 50,
                 position_unstim: int = 52,
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
        torch.tensor([50]),
    ]

    ################
    # Run experiment
    ################

    ys = []
    y_errs = []
    from math import log

    for ns_fraction in np.linspace(0.1,1,10):

        # Introduce FK506
        model_p_dict = model_p_dict_orig.copy()
        model_p_dict['Ns'] = model_p_dict['Ns']*ns_fraction
        model_p_dict['mu_log_K_N'][1] = model_p_dict['mu_log_K_N'][1] + log(ns_fraction)


        # Simulation

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

            # simulation.visualize_run(1,2)

            basal_sizes, sizes, rel_sizes = simulation.run(100)

            basal_size_batch.append(basal_sizes.squeeze())
            size_batch.append(sizes.squeeze())
            rel_size_batch.append(rel_sizes.squeeze())

        rsb = rel_size_batch[0]
        sem_rs, avg_rs = torch.std_mean(rsb, axis=2)


        x_post = torch.arange(2,20)

        y_unstim = avg_rs[x_post, position_unstim]*100
        yerr_unstim = sem_rs[x_post, position_unstim]*100

        ys.append(y_unstim)
        y_errs.append(yerr_unstim)



    ###########
    # Plotting
    ###########
    x_pre = torch.arange(-10,1)
    y_pre = torch.ones(len(x_pre))*100

    if len(axs) == 0:
        fig, axs = plt.subplots(1,2,figsize=(8,3), dpi=100)
        fig.subplots_adjust(left=0.1, bottom=0.35,right=0.98,top=0.95,
                            wspace=0.4)
    
    rdngreen = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

    # Simulation
    ax = axs[0]

    ax.errorbar(x_pre, y_pre, c='tab:blue', fmt='.')
    ax.plot(x_pre, y_pre, c='tab:blue', alpha=0.2)

    for y in ys:
        # ax.plot(x_post, y, alpha=0.2, label='Control',
        #         c='tab:blue')
        ax.plot(x_post, y, alpha=0.2)
    # ax.errorbar(x_post, ys[1], yerr=y_errs[1], fmt='.', 
    #             markersize=9, label=r'$N_s$ halfed',
    #             c='tab:blue')


    ax.axhline(y=100, linewidth=1, linestyle=(1,(5,5)))
    ax.arrow(1,115, 0, -5, head_width=1, head_length=5, width=0.2, color=rdngreen)
    # ax.set_ylim(80,120)
    
    ax.set_xlabel('Time [min]')
    ax.set_ylabel(r'$P/P_{pre}$ (%)')
    ax.legend(frameon=False, loc='lower left')

    # Paper data
    data = np.genfromtxt('data/validation_data/scanziani_data/microcystine.csv',
                         delimiter=',').T

    data_control = np.genfromtxt('data/validation_data/scanziani_data/unstim.csv',
                         delimiter=',').T
    
    ax = axs[1]
    ax.errorbar(data[0], data[1], data[2],
                fmt='.', markersize=9, c='black', elinewidth=1,
                label='Microcystine')
    ax.plot(data_control[0], data_control[1], alpha=0.2, color='black', label='Control')
    ax.axhline(y=100, linewidth=1, linestyle=(1,(5,5)), c='black')
    ax.arrow(0.2,120, 0, -10, head_width=1, head_length=5, width=0.2, color='black')

    ax.set_ylim(60,120)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('E.p.s.c. (%)')
    ax.legend(frameon=False, loc='lower left')

    if save_folder:
        plt.savefig(join(save_folder, 'sim_scanziani_mc.png'))

    plt.show()
