
import torch
import matplotlib.pyplot as plt
from scipy.stats import lognorm

from .. import Simulation




def Test(model, model_p_dict):

    ################
    # Control panel
    ################

    simulation_time = 40
    inter_spine_distance = 1
    
    spine_number = 1000
    stim_idxes = torch.arange(1,2,1)


    ################
    # Run experiment
    ################

    simulation = Simulation(model = model,
                            model_p_dict = model_p_dict,
                            simulation_time = simulation_time,
                            spine_number = spine_number,
                            inter_spine_distance = inter_spine_distance,
                            stim_indexes = stim_idxes)

    bsb, sb, rsb = [srs[2:] for srs in simulation.run(100)]

    y = bsb[0,:,0]
    fitres = lognorm.fit(y, floc=0)

    fig, ax = plt.subplots(1,1,figsize=(4,3), dpi=300)
    fig.tight_layout(rect=[0.1,0.1,1,1])

    _, bins, _ = ax.hist(bsb[0,:,0], density=True, color='tab:blue', 
                         label='simulated data', bins=50)
    ax.plot(bins, lognorm.pdf(bins, *fitres), color='tab:orange', 
            linewidth=3, label='lognormal fit')
    
    ax.ticklabel_format(axis='x', style='plain')

    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel('spine P', fontsize=10)
    ax.set_ylabel('pdf', fontsize=10)
    plt.savefig('lognorm_P.png')
    plt.show()

