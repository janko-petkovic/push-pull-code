from os.path import join
import torch
torch.manual_seed(2023)

import matplotlib.pyplot as plt
import numpy as np

from .. import Simulation


def rojer_mexican(model, 
                  model_p_dict, 
                  simulation_time: int = 40,
                  spine_number: int = 1000,
                  inter_spine_distance: int = 1,
                  ax = None,
                  save_folder: str = ''):

    ################
    # Control panel
    ################


    stim_configurations = [
        torch.tensor([50]) 
    ]

    ######## Plotting variables
    position_idx = torch.arange(50, 60)
    time_idx = 2

    data = np.genfromtxt('data/validation_data/royer_data/data.csv', delimiter=',').T

    ################
    # Run experiment
    ################

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

        # simulation.visualize_run(10,2)

        bsb, sb, _ = simulation.run(100)

    ndb = (sb - bsb)/bsb
    sem_rs, avg_rs = torch.std_mean(ndb, axis=2)


    ######## Plotting
    # Default axes configuration if axes are not provided
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(4,3), dpi=150)

    x = torch.arange(10)
    y = avg_rs[time_idx,position_idx]
    yerr = sem_rs[time_idx,position_idx]

    # Main plot
    ax.plot(x,y, linewidth=3)
    ax.errorbar(data[0], data[1], data[2], c='black', fmt='.', markersize=8,
                capsize=3, elinewidth=1)
    ax.axhline(y=0, linestyle='--', linewidth=1)

    ax.set_xlabel(r'Distance from LTP site [$\mu m$]')
    ax.set_ylabel(r'Normalized post-pre $P$')

    # Inset
    position_idx = torch.arange(50, 100)
    time_idx = 2

    x = torch.arange(50)
    y = avg_rs[time_idx,position_idx]

    # ax_ins.plot(x,y)
    # ax_ins.axhline(y=0, linestyle='--', linewidth=1)


    # if save_folder:
    #     plt.savefig(join(save_folder, 'sim_royer.png'))


