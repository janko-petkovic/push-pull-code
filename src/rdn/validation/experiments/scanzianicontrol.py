from os.path import join
import matplotlib.pyplot as plt
import torch
torch.manual_seed(2023)
import numpy as np

from .. import Simulation


def scanziani_control(model, 
                      model_p_dict,
                      simulation_time: int = 40,
                      spine_number: int = 100,
                      position_stim: int = 50,
                      position_unstim: int = 54,
                      inter_spine_distance: int = 1,
                      axs: list = [],
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


    ###########
    # Plotting
    ###########

    # We want percentages
    x_post = torch.arange(2,20)
    y_stim = avg_rs[x_post, position_stim]*100
    yerr_stim = sem_rs[x_post, position_stim]*100

    y_unstim = avg_rs[x_post, position_unstim]*100
    yerr_unstim = sem_rs[x_post, position_unstim]*100

    x_pre = torch.arange(-10,1)
    y_pre = torch.ones(len(x_pre))*100

    if len(axs)==0:
        fig, axs = plt.subplots(2,2, figsize=(8,3), dpi=100)
        fig.subplots_adjust(left=0.1, bottom=0.2, right=0.98, top=0.98,
                            wspace=0.4)

    rdngreen = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    rdnorange = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]

    # Stimulated
    ax = axs[0,0]
    ax.errorbar(x_post, y_stim, yerr=yerr_stim,
                fmt='^', c ="white", ecolor=rdngreen, elinewidth=1,
                mew=1, mec=rdngreen, markersize=7)
    ax.errorbar(x_pre, y_pre,
                fmt='^', markerfacecolor ="white", mec=rdngreen, 
                mew=1, label='homo', markersize=7)
    
    ax.arrow(1,130, 0, -10, head_width=1, head_length=10, width=0.2,
             color=rdngreen)    
    ax.set_ylim(80,(y_stim+yerr_stim).max()*1.1)
    ax.legend(loc='upper left')

    # Unstimulated
    ax = axs[1,0]
    ax.errorbar(x_post, y_unstim, yerr=yerr_unstim,
        fmt='.', markersize=9, c=rdngreen, ecolor=rdngreen,
    )
    ax.errorbar(x_pre, y_pre, 
        fmt='.', markersize=9, label=r'hetero'
    )
    ax.arrow(1,107, 0, -2, head_width=1, head_length=2, width=0.2,
             color=rdngreen)    
    
    ax.set_ylim((y_unstim- yerr_unstim).min()*0.9, 120)
    ax.legend(frameon=False, loc='lower left')


    ##### Scanziani data
    stim_data = np.genfromtxt('data/validation_data/scanziani_data/stim.csv', delimiter=',').T
    unstim_data = np.genfromtxt('data/validation_data/scanziani_data/unstim.csv', delimiter=',').T
    # I correct the stim data cos of wrong digitization
    stim_data[1] = (stim_data[1]-100)* 50/15 + 100
    stim_data[2] *= 50/15

    # Stim
    ax = axs[0,1]
    ax.errorbar(stim_data[0], stim_data[1], stim_data[2],
                fmt='^', markersize=7, mew=1, 
                markerfacecolor='white', ecolor='black', mec='black',
                label='homo')
    ax.arrow(1,150, 0, -10, head_width=1, head_length=10, width=0.2,
             color='black')    
    ax.set_ylim(80, (stim_data[1] + stim_data[2]).max()*1.1)
    ax.legend(loc='upper left')

    ax = axs[1,1]
    ax.errorbar(unstim_data[0], unstim_data[1], unstim_data[2],
                fmt='.', markersize=9, c='black', label='hetero')
    ax.set_ylim((unstim_data[1] - unstim_data[2]).min()*0.9, 120)
    ax.arrow(1,115, 0, -5, head_width=1, head_length=5, width=0.2,
             color='black')
    ax.legend(loc='lower left')

    for ax in axs[:,0]:
        ax.set_ylabel(r'$P / P_{pre}$ (%)')
        ax.axhline(y=100, linewidth=1, linestyle=(1,(5,5)))

    for ax in axs[:,1]:
        ax.set_ylabel('Field\ne.p.s.p. (%)')
        ax.axhline(y=100, linestyle=(1,(5,5)), linewidth=1, c='black')
    
    for ax in axs[1]:
        ax.set_xlabel('Time [min]')

    if save_folder:
        plt.savefig(join(save_folder, 'sim_scanziani_control.png'))

    plt.show()
