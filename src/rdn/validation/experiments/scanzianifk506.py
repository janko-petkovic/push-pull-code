
import torch
torch.manual_seed(2023)

from math import sqrt, exp
import matplotlib.pyplot as plt
import scipy.stats as stats

from .. import Simulation




def scanziani_mc(model, model_p_dict):

    ################
    # Control panel
    ################

    simulation_time = 40
    spine_number = 1000
    inter_spine_distance = 1

    stim_configurations = [
        # torch.tensor([8,9,10]),
        # torch.tensor([9,10]),
        # torch.tensor([8,10]),
        torch.tensor([50]),
    ]

    ################
    # Run experiment
    ################

    # model_p_dict['mu_log_K'] *= 1.3
    # model_p_dict['Ks'] /= 3
    model_p_dict['Ns'] = 0

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

        basal_sizes, sizes, rel_sizes = simulation.run(10)

        basal_size_batch.append(basal_sizes.squeeze())
        size_batch.append(sizes.squeeze())
        rel_size_batch.append(rel_sizes.squeeze())

    bsb = basal_size_batch[0]
    sb = size_batch[0]
    rsb = rel_size_batch[0]

    # CAREFUL HERE
    avg_rs = rsb.mean(axis=2)
    sem_rs = rsb.std(axis=2)/sqrt(len(rsb[0,0]))*3


    ###########
    # Plotting
    ###########
    position_stim = 50
    position_unstim = 57

    x_post = torch.arange(2,20)
    y_stim = avg_rs[x_post, position_stim]
    yerr_stim = sem_rs[x_post, position_stim]

    y_unstim = avg_rs[x_post, position_unstim]
    yerr_unstim = sem_rs[x_post, position_unstim]

    x_pre = torch.arange(-10,1)
    y_pre = torch.ones(len(x_pre))



    fig, ax = plt.subplots(1,1,figsize=(4,1.6), dpi=300)
    fig.tight_layout(rect=[0.07,0.1,0.95,1])

    ax.errorbar(x_post, y_unstim, yerr=yerr_unstim,
        fmt='.')
    ax.errorbar(x_pre, y_pre,
        fmt='.', c='tab:blue', label='stim')
    
    
    
    ax.set_ylim(0.6,1.2)
    
    ax.set_xlabel('Time [min]')
    ax.set_ylabel(r'$P/P_{pre}$')

    plt.legend()


    plt.savefig('/home/janko/code/phd/rdn-project/docs/paper/src/figures/results/sim_scanziani_mc.png')

    plt.show()