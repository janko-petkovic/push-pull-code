import torch 
torch.manual_seed(2)

import numpy as np
np.random.seed(2)

import pandas as pd
import matplotlib.pyplot as plt

from ...defaults import default_goda_transformation, default_goda_filter
from ...dataloading import df_from_tool
from ...fitting.models import BaseModel

from .. import Simulation




def GodaSevenClust(model: BaseModel,
                   model_p_dict: dict, 
                   data_path: str,
                   bins: list,
                   t_idx: int = 2,
                  ) -> None:


    df = df_from_tool(
            root=data_path,
            dataset='goda',
            compartment='spine'
        )

    df = df[df['drug'] == 'Control']
    df = default_goda_transformation(df)
    df = default_goda_filter(df)
    

    ##################
    # Binning the experimental behaviour
    ##################

    # Note that here we select the nss
    dff = df[(df['nss'] == 7) & (df['distance'] < 15)]

    # Take out the stimulated statistics
    sdff = dff[dff['type']=='Stim'][f'norm_{t_idx}']
    s_median = sdff.median()
    s_q1 = sdff.quantile(q=0.25)
    s_q3 = sdff.quantile(q=0.75)


    # Now the unstim binning
    dff = dff[dff['type']=='Spine']
    dff['Y'] = dff['norm_2']

    dff['bin'] = pd.cut(dff['distance'], bins)
    tdff = dff[['bin','Y']]
    b_dff = dff.groupby('bin').agg({'distance' : 'mean', 'Y' : ['median']})
    b_dff['Y','q1'] = tdff.groupby('bin').quantile(q=0.25)
    b_dff['Y','q3'] = tdff.groupby('bin').quantile(q=0.75)
    b_dff.reset_index(drop=True, inplace=True)

    # Add the stimulated bin again
    b_dff.loc[10] = [0, s_median, s_q1, s_q3]

    # Take the plotting data
    X_data = b_dff['distance', 'mean']
    Y_data = b_dff['Y', 'median']
    Y_q1_data = b_dff['Y', 'q1']
    Y_q3_data = b_dff['Y', 'q3']

    Y_el_data = Y_data-Y_q1_data
    Y_eh_data = Y_q3_data-Y_data

    # plt.errorbar(X_data,Y_data, yerr=(Y_el_data, Y_eh_data), fmt='.')


    ##############
    # SIMULATION #
    ##############

    # Inputs for the simulation
    simulation_time = 40
    spine_number = 1000
    inter_spine_distance = 1
    n_experiments = 200
    stim_configurations = [torch.arange(110,145,5)]

    # Run simulation
    rel_sizes_batch = None

    for stim_idxes in stim_configurations:
        simulation = Simulation(model = model,
                                model_p_dict = model_p_dict,
                                simulation_time = simulation_time,
                                spine_number = spine_number,
                                inter_spine_distance = inter_spine_distance,
                                stim_indexes = stim_idxes)

        _, _, rel_sizes_batch = simulation.run(n_experiments)

    # Range we take for the visualization
    lr = 2 
    ur = 13
    stim_idx = 140
    
    # Extract data for plotting
    Y_batch = rel_sizes_batch[t_idx, stim_idx-lr:stim_idx+ur]
    Y, _ = Y_batch.median(axis=1)
    Y_1q = torch.quantile(Y_batch, 0.25, axis=1)
    Y_3q = torch.quantile(Y_batch, 0.75, axis=1)

     
    #################
    # VISUALIZATION #
    #################
    X = np.arange(-lr, ur)

    fig = plt.figure(figsize=(7,3), dpi=300)
    gs = plt.GridSpec(2,2)

    ## Left panel
    ax = fig.add_subplot(gs[:,0])

    # Data
    ax.errorbar(X_data,
                Y_data,
                yerr=(Y_el_data, Y_eh_data),
                fmt='.', c='black', markersize=10, capsize=2, linewidth=1,
                alpha=1, label='Data (RID)'
                )

    # Prediction
    ax.plot(X, Y, linewidth=3, label='Model (P)')
    ax.fill_between(X, Y_1q, Y_3q, alpha=0.1)

    # Baseline
    ax.axhline(y=1, linestyle='--', linewidth=1, c='black')

    ax.set_ylim(0.5,2)
    ax.set_xlabel(r'Distance from stimulation [$\mu m$]')
    ax.set_ylabel(f'Normalized value at t = {t_idx}')
    ax.legend(frameon=False)


    ## Right panel, upper
    ax = fig.add_subplot(gs[0,1])
    ax.hist(rel_sizes_batch[t_idx, stim_idx,:].flatten(), bins=50, density=True,
            label='stim P density')
    ax.set_xlim(0,4)
    ax.set_xlabel(f'Normalized stim P, $t = {t_idx}$ min')
    ax.set_ylabel('pdf')
    # ax.legend(frameon=False, fontsize=8)

    ## Right panel, lower
    ax = fig.add_subplot(gs[1,1])
    sdff.hist(ax=ax, bins=20, color='black', grid=False, density=True,
              label='stim RID density')

    ax.set_xlim(0,4)
    ax.set_xlabel(f'Normalized stim RID, $t = {t_idx}$ min')
    ax.set_ylabel('pdf')
    # ax.legend(frameon=False, fontsize=8)

    fig.tight_layout(pad=1)


    plt.show()

    

