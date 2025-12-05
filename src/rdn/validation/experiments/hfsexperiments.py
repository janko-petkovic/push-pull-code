from os.path import join
import torch
torch.manual_seed(2023)

import matplotlib.pyplot as plt
import numpy as np

from .. import Simulation



def hfs_experiments(model,
                    model_p_dict,
                    simulation_time: int = 40,
                    spine_number: int = 1000,
                    inter_spine_distance: int = 1,
                    axs: list  = [],
                    save_folder: str = ''):

    ################
    # Control panel
    ################

    stim_idx = 50
    ur = 10

    stim_configurations = [
        torch.tensor([50]),
    ]

    ###############
    # Import data #
    ###############

    royer_data = np.genfromtxt(
            'data/validation_data/royer_data/data.csv',
            delimiter=',').T
    
    scan_stim_data = np.genfromtxt(
            'data/validation_data/scanziani_data/stim.csv',
            delimiter=',').T
    scan_unstim_data = np.genfromtxt(
            'data/validation_data/scanziani_data/unstim.csv',
            delimiter=',').T

    ################
    # Run experiment
    ################

    basal_size_batch = []
    size_batch = []
    rel_size_batch = []
    
    # Baseline experiment
    for stim_idxes in stim_configurations:
        simulation = Simulation(model = model,
                                model_p_dict = model_p_dict,
                                simulation_time = simulation_time,
                                spine_number = spine_number,
                                inter_spine_distance = inter_spine_distance,
                                stim_indexes = stim_idxes)

        *_, rel_sizes = simulation.run(100)

    sem_rs, avg_rs = torch.std_mean(rel_sizes, axis=2) #pyright: ignore
    avg_delta = avg_rs - 1

    # Modified royer
    royer_model_p_dict = model_p_dict.copy()
    fK = 2
    fN = 10
    royer_model_p_dict['sigma_N'] = royer_model_p_dict['sigma_N']*fN
    royer_model_p_dict['sigma_K'] = royer_model_p_dict['sigma_K']*fK
    royer_model_p_dict['Ks'] = royer_model_p_dict['Ks']/fK/3
    royer_model_p_dict['Ns'] = royer_model_p_dict['Ns']/fN
    royer_model_p_dict['tau_K'] = royer_model_p_dict['tau_K']*5
    royer_model_p_dict['tau_N'] = royer_model_p_dict['tau_N']*5

        
    for stim_idxes in stim_configurations:
        simulation = Simulation(model = model,
                                model_p_dict = royer_model_p_dict,
                                simulation_time = simulation_time,
                                spine_number = spine_number,
                                inter_spine_distance = inter_spine_distance,
                                stim_indexes = stim_idxes)

        basal_sizes, _, rel_sizes = simulation.run(100)

    ub_sizes = basal_sizes.median(axis=1)[0][2]*1.4
    lb_sizes = basal_sizes[2].quantile(q=0.05, axis=0)
    
    ub_idxes = torch.where(basal_sizes[2, stim_idx] < ub_sizes)[0]
    lb_idxes = torch.where(basal_sizes[2, stim_idx] > lb_sizes)[0]
    good_idxes = torch.tensor(np.intersect1d(ub_idxes, lb_idxes))

    rel_sizes = rel_sizes[:,:,good_idxes]
    sem_rrs, avg_rrs = torch.std_mean(rel_sizes, axis=2)
    avg_rdelta = avg_rrs - 1

    # Modified scanziani
    scanziani_model_p_dict = model_p_dict.copy()
    fK = 2
    fN = 10
    scanziani_model_p_dict['sigma_N'] = scanziani_model_p_dict['sigma_N']*fN
    scanziani_model_p_dict['sigma_K'] = scanziani_model_p_dict['sigma_K']*fK
    scanziani_model_p_dict['Ks'] = scanziani_model_p_dict['Ks']/fK/5.5
    scanziani_model_p_dict['Ns'] = scanziani_model_p_dict['Ns']/fN
    scanziani_model_p_dict['tau_K'] = scanziani_model_p_dict['tau_K']*5
    scanziani_model_p_dict['tau_N'] = scanziani_model_p_dict['tau_N']*5

        
    for stim_idxes in stim_configurations:
        simulation = Simulation(model = model,
                                model_p_dict = scanziani_model_p_dict,
                                simulation_time = simulation_time,
                                spine_number = spine_number,
                                inter_spine_distance = inter_spine_distance,
                                stim_indexes = stim_idxes)

        basal_sizes, _, rel_sizes = simulation.run(100)


    ub_sizes = basal_sizes.median(axis=1)[0][2]*0.8
    lb_sizes = basal_sizes[2].quantile(q=0.05, axis=0)
    
    ub_idxes = torch.where(basal_sizes[2, stim_idx] < ub_sizes)[0]
    lb_idxes = torch.where(basal_sizes[2, stim_idx] > lb_sizes)[0]
    good_idxes = torch.tensor(np.intersect1d(ub_idxes, lb_idxes))

    rel_sizes = rel_sizes[:,:,good_idxes]
    sem_srs, avg_srs = torch.std_mean(rel_sizes, axis=2)
    avg_sdelta = avg_srs-1


    # Modified onuma
    onuma_model_p_dict = model_p_dict.copy()
    fK = 2
    fN = 10
    onuma_model_p_dict['sigma_N'] = onuma_model_p_dict['sigma_N']*fN
    onuma_model_p_dict['sigma_K'] = onuma_model_p_dict['sigma_K']*fK
    onuma_model_p_dict['Ks'] = onuma_model_p_dict['Ks']/fK/5.5
    onuma_model_p_dict['Ns'] = onuma_model_p_dict['Ns']/fN/2
    onuma_model_p_dict['tau_K'] = onuma_model_p_dict['tau_K']*5
    onuma_model_p_dict['tau_N'] = onuma_model_p_dict['tau_N']*5
    onuma_model_p_dict['mu_log_K_N'][1] /= 2


        
    for stim_idxes in stim_configurations:
        simulation = Simulation(model = model,
                                model_p_dict = scanziani_model_p_dict,
                                simulation_time = simulation_time,
                                spine_number = spine_number,
                                inter_spine_distance = inter_spine_distance,
                                stim_indexes = stim_idxes)

        basal_sizes, _, rel_sizes = simulation.run(100)


    ub_sizes = basal_sizes.median(axis=1)[0][2]*0.8
    lb_sizes = basal_sizes[2].quantile(q=0.05, axis=0)
    
    ub_idxes = torch.where(basal_sizes[2, stim_idx] < ub_sizes)[0]
    lb_idxes = torch.where(basal_sizes[2, stim_idx] > lb_sizes)[0]
    good_idxes = torch.tensor(np.intersect1d(ub_idxes, lb_idxes))

    rel_sizes = rel_sizes[:,:,good_idxes]
    sem_ors, avg_ors = torch.std_mean(rel_sizes, axis=2)
    avg_odelta = avg_ors-1

    ############
    # PLOTTING #
    ############

    t_idx = 40

    if len(axs)==0:
        fig, axs = plt.subplots(2,3)



    ax = axs[0,0]
    X = np.arange(ur)
    Y = avg_delta[t_idx][stim_idx:stim_idx+ur]
    # Y_err = sem_rs[2][stim_idx:stim_idx+ur]

    rY = avg_rdelta[t_idx][stim_idx:stim_idx+ur]
    rY_err = sem_rrs[t_idx][stim_idx:stim_idx+ur]
    ax.plot(X, Y)
    ax.errorbar(X, rY, rY_err)
    

    ax = axs[1,0]
    ax.errorbar(royer_data[0], royer_data[1], royer_data[2], c='black',
                fmt='.', markersize=8, capsize=3, elinewidth=1)
    ax.fill_between(royer_data[0], royer_data[1], color='black')
    ax.axhline(y=0, linestyle='--', linewidth=1, c='black')

    ax.set_xlabel('Distance from LTP site')
    ax.set_ylabel('Normalized post-pre')


    ax = axs[0,1]
    X_spre = np.arange(-10, 0)[::4]
    X_upre = np.arange(-10, 0)[::4]+2
    X_post = np.arange(2,20)[::4]
    
    Y_spost = avg_sdelta[X_post,stim_idx]
    Y_err_spost = sem_srs[X_post, stim_idx]

    Y_upost = avg_sdelta[X_post,stim_idx+5]
    Y_err_upost = sem_srs[X_post, stim_idx+5]

    ax.errorbar(X_post, Y_spost, Y_err_spost)
    ax.errorbar(X_post, Y_upost, Y_err_upost)
    ax.scatter(X_spre, [0.]*len(X_spre))
    ax.scatter(X_upre, [0.]*len(X_upre))

    ax = axs[1,1]
    ax.errorbar(
            scan_stim_data[0],
            scan_stim_data[1],
            scan_stim_data[2],
            )

    ax.errorbar(
            scan_unstim_data[0],
            scan_unstim_data[1],
            scan_unstim_data[2],
            )

    ax = axs[0,2]

    ax = axs[1,2]



    # ######## Plotting variables
    # position_idx = torch.arange(50, 60)
    # time_idx = 2
    #
    # x = torch.arange(10)
    # y = avg_rs[time_idx,position_idx]
    # yerr = sem_rs[time_idx,position_idx]

    plt.show()

    # ######## Plotting
    # # Default axes configuration if axes are not provided
    # if len(axs)==0:
    #     fig, axs = plt.subplots(1,2, figsize=(8,3), dpi=150)
    #     ax_ins = fig.add_axes([0.3,0.6,0.15,0.3]) # For the inset
    #     fig.subplots_adjust(left=0.1, bottom=0.2, right=0.98, top=0.98,
    #                         wspace=0.4)
    # else:
    #     ax, ax3, ax_ins = axs


    # if save_folder:
    #     plt.savefig(join(save_folder, 'sim_royer.png'))


    # plt.show()
