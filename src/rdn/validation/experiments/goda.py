from os.path import join
import torch
torch.manual_seed(2023)

from math import sqrt
import matplotlib.pyplot as plt

from .. import Simulation




def Goda2022(model, model_p_dict, save_path=None):

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

        close_base_RID_per_nss.append(batch[t_idx][stim_config[0]-r:stim_config[-1]+r:2])


    # AFTER STIMULUS:
    # RID
    tot_RID_per_nss = []
    close_RID_per_nss = []
    stim_RID_per_nss = []

    for batch, stim_config in zip(exp_size_batches, stim_configurations):
        tot_RID_per_nss.append(batch[t_idx])
        stim_RID_per_nss.append(batch[t_idx][stim_config])
        close_RID_per_nss.append(batch[t_idx][stim_config[0]-r:stim_config[-1]+r:2])


    # Delta
    close_delta_per_nss = []
    tot_delta_per_nss = []
    stim_delta_per_nss = []

    for bsb, sb, stim_config in zip(exp_basal_size_batches, exp_size_batches, stim_configurations):
        tot_delta_per_nss.append(
            sb[t_idx] - bsb[t_idx]
        )
            
        close_delta_per_nss.append(
            sb[t_idx][stim_config[0]-r:stim_config[-1]+r:2] - bsb[0][stim_config[0]-r:stim_config[-1]+r:2]
        )

        stim_delta_per_nss.append(
            sb[t_idx][stim_config] - bsb[t_idx][stim_config]
        )


    # Norm
    close_norm_per_nss = []
    tot_norm_per_nss = []
    stim_norm_per_nss = []

    for batch, stim_config in zip(exp_rel_size_batches, stim_configurations):
        tot_norm_per_nss.append(batch[t_idx])
        stim_norm_per_nss.append(batch[t_idx][stim_config])


    ###########
    # Plotting
    ###########
    def plot_statistic(data, ax):
        means = [d.mean() for d in data]
        errs = [d.std()/sqrt(len(d.flatten())) for d in data]
        ax.errorbar(nsss, means, yerr=errs)
        ax.set_xticks(nsss)
        ax.set_xticklabels(nsss)


    fig, axs = plt.subplots(3,2,figsize=(8,8), dpi=300)
    fig.tight_layout(pad=5)


    # RID
    ax = axs[0,0]
    # plot_statistic(tot_base_RID_per_nss, ax)
    # plot_statistic(tot_RID_per_nss, ax)
    plot_statistic(close_base_RID_per_nss, ax)
    plot_statistic(close_RID_per_nss, ax)

    ax.set_xlabel('# Stimulations')
    ax.set_ylabel('P');
    ax.set_title('Tot size per nss');


    ax = axs[0,1]
    plot_statistic(stim_base_RID_per_nss, ax)
    plot_statistic(stim_RID_per_nss, ax)

    ax.set_xlabel('# Stimulations')
    ax.set_ylabel('P');
    ax.set_title('Tot stim size per nss');


    # Delta
    ax = axs[1,0]
    # plot_statistic(tot_delta_per_nss, ax)
    plot_statistic(close_delta_per_nss, ax)

    ax.set_xlabel('# Stimulations')
    ax.set_ylabel('P');
    ax.set_title('Tot delta per nss');

    ax = axs[1,1]
    plot_statistic(stim_delta_per_nss, ax)

    ax.set_xlabel('# Stimulations')
    ax.set_ylabel('norm P');
    ax.set_title('Stim delta per nss');


    # Norm
    ax = axs[2,0]
    plot_statistic(tot_norm_per_nss, ax)

    ax.set_xlabel('# Stimulations')
    ax.set_ylabel('P');
    ax.set_title('Tot norm per nss');

    ax = axs[2,1]
    plot_statistic(stim_norm_per_nss, ax)

    ax.set_xlabel('# Stimulations')
    ax.set_ylabel('norm P');
    ax.set_title('Stim norm per nss');

    if save_path:
        plt.savefig(join(save_path, 'goda_tot_vs_stim.png'))

    plt.show()
