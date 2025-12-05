from os.path import join
import torch 
torch.manual_seed(2)

import numpy as np
np.random.seed(2)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ...defaults import default_goda_transformation, default_goda_filter
from ...dataloading import df_from_tool
from ...fitting.models import BaseModel

from .. import Simulation


def get_plot_series_from_df(df, bins, t_idx):
    '''Auxiliary function to get the data to plot from the experimental
    dataframe
    '''


    dff = df[(df['nss'] == -1) & (df['distance'] < 15)]

    # Take out the stimulated statistics
    sdff = dff[dff['type']=='Stim'][f'norm_{t_idx}']
    s_median = sdff.median()
    s_q1 = sdff.quantile(q=0.25)
    s_q3 = sdff.quantile(q=0.75)


    # Now the unstim binning
    dff = dff[dff['type']=='Spine']
    dff['Y'] = dff[f'norm_{t_idx}']

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

    return X_data, Y_data, Y_el_data, Y_eh_data


def get_plot_series_from_simulation(rel_sizes_batch, lr, ur, stim_idx, t_idx):
    '''Auxiliary function to get the data to plot from the experimental
    dataframe.
    Notice that it returns not the errors but the 1 and 3 quantiles, since they
    will be used with fillbetween
    '''

    
    # Extract data for plotting
    X = np.arange(-lr, ur)
    Y_batch = rel_sizes_batch[t_idx, stim_idx-lr:stim_idx+ur]
    Y, _ = Y_batch.median(axis=1)
    Y_1q = torch.quantile(Y_batch, 0.25, axis=1)
    Y_3q = torch.quantile(Y_batch, 0.75, axis=1)

    return X, Y, Y_1q, Y_3q


# Transform in percentages
def ttp(*data):
    '''transform to percentages'''
    res = []
    for d in data:
        res.append(d*100)
    return res

def ttc(*data):
    '''transform to changes'''
    res = []
    for d in data:
        res.append(d-1.)
    return res

def plot_prediction_plot(ax, df, rel_sizes_batch, bins, lr, ur, stim_idx, t_idx,
                         **kwargs):
    '''The main plot function (plots the classic prediction vs data)'''

    # Pure data
    dff = df[(df['nss'] == -1) & (df['distance'] < 15)]
    ax.scatter(dff['distance'], dff[f'norm_{t_idx}']-1, alpha=0.1, c='gray',
               linewidths=0, s=kwargs.get('s'),)


    # Binned data
    X_data, Y_data, Y_el_data, Y_eh_data = get_plot_series_from_df(df, bins, t_idx) 

    Y_data, _ = ttc(Y_data,1)

    ax.errorbar(X_data,
                Y_data,
                yerr=(Y_el_data, Y_eh_data),
                fmt='.', 
                c='black',
                markersize=kwargs.get('markersize'),
                capsize=kwargs.get('capsize'),
                linewidth=1,
                label=r'Experiment',
                zorder=100,
                )
                

    # Prediction
    X, Y, Y_1q, Y_3q = get_plot_series_from_simulation(rel_sizes_batch,
                                                       lr,
                                                       ur,
                                                       stim_idx, 
                                                       t_idx)
    Y, Y_1q, Y_3q = ttc(Y, Y_1q, Y_3q)

    ax.plot(X, Y, linewidth=kwargs.get('linewidth'),
            label=r'Prediction')
    ax.fill_between(X, Y_1q, Y_3q, alpha=0.1)
    ax.axhline(y=0, linewidth=1, linestyle=(0,(5,5)), c='black')


def plot_distribution_plot(axs, df, rel_sizes_batch, stim_idx, t_idx):

    bins = 20

    ## Right panel, upper
    ax = axs[0]
    ax.hist(rel_sizes_batch[t_idx, stim_idx,:].flatten()-1, 
            bins=bins,
            density=True,
            label='Prediction',
            )

    ## Right panel, lower
    ax = axs[1]
    dff = df[(df['nss'] == -1) & (df['distance'] < 15)]
    sdff = dff[dff['type']=='Stim'][f'norm_{t_idx}']

    Y = (sdff-1).to_numpy()

    ax.hist(Y, bins=bins, density=True, color='black', align='right',
            label='Experiment',)


def goda_seven_distr(model: BaseModel, 
                     model_p_dict: dict, 
                     simulation_time: int = 40,
                     spine_number: int = 1000,
                     inter_spine_distance: int = 1,
                     bins: list = [-15,-10,-5,0,5,10,15],
                     axs: list = [],
                     save_folder: str = '') -> None:

    ##############
    # PARAMETERS #
    ##############
    t_idxes = [2, 20, 30, 40]
    n_experiments = 100
    stim_configurations = [torch.arange(110,390,40)]

    # Range we take for the visualization
    lr = 11
    ur = 15

    # Stimulated index for the visualization
    stim_idx = 350


    ################
    # DATA LOADING #
    ################

    # Import the experimental dataset
    df = df_from_tool(
            root='data/TomData',
            dataset='goda',
            compartment='spine'
        )
    
    # Select the control condition and apply the default filtering
    df = df[df['drug'] == 'Control']
    df = default_goda_transformation(df)
    df = default_goda_filter(df)
    

    ##############
    # SIMULATION #
    ##############

    rel_sizes_batch = None

    for stim_idxes in stim_configurations:
        simulation = Simulation(model = model,
                                model_p_dict = model_p_dict,
                                simulation_time = simulation_time,
                                spine_number = spine_number,
                                inter_spine_distance = inter_spine_distance,
                                stim_indexes = stim_idxes)

        _, _, rel_sizes_batch = simulation.run(n_experiments)


    #################
    # VISUALIZATION #
    #################

    if len(axs) == 0:
        fig = plt.figure(figsize=(12,3), dpi=100)
        gs = fig.add_gridspec(6,5, width_ratios=(0.5,0.3,1,0.3,0.5))
        fig.subplots_adjust(wspace=0, bottom=0.2, left=0.02, right=0.94, top=0.8)

        axs = []
        axu = fig.add_subplot(gs[0:3,0])
        axd = fig.add_subplot(gs[3:,0])
        axs += [axu, axd]
        axs.append(fig.add_subplot(gs[:,2]))
        
        axs += [
                fig.add_subplot(gs[:2,4]),
                fig.add_subplot(gs[2:4,4]),
                fig.add_subplot(gs[4:,4]),
                ]


    distr_axs = axs[:2]
    plot_distribution_plot(distr_axs, df, rel_sizes_batch, stim_idx, 2)

    center_ax = axs[2]
    plot_prediction_plot(center_ax, df, rel_sizes_batch, bins, lr, ur, stim_idx,
                         2, linewidth=3, markersize=10, capsize=2, s=10)

    right_axs = axs[3:]
    plot_prediction_plot(right_axs[0], df, rel_sizes_batch, bins, lr, ur,
                         stim_idx, 10, linewidth=2, markersize=4, capsize=0,s=3)

    plot_prediction_plot(right_axs[1], df, rel_sizes_batch, bins, lr, ur,
                         stim_idx, 30, linewidth=2, markersize=4, capsize=0,
                         s=3)

    plot_prediction_plot(right_axs[2], df, rel_sizes_batch, bins, lr, ur,
                         stim_idx, 40, linewidth=2, markersize=4, capsize=0,
                         s=3)

    # Cosmetics
    # Distributions
    distr_axs[0].set_xticklabels([])
    distr_axs[1].set_xlabel('Stimulated relative\nchange (2 min)', size=10,
                            labelpad=3,)
    for ax in distr_axs:
        ax.set_ylabel('pdf', size=10, labelpad=5)
        ax.set_xlim(-1,3)
        ax.set_xticks([-1,0,3])
        ax.set_ymargin(.8)
        ax.set_yticks([])
        ax.legend(loc='upper left', fontsize=10)
    

    # Central ax
    ax = center_ax
    ax.legend(loc='upper right')
    ax.text(-10,.6,r'2 minutes', size=13)
    ax.set_xlim(-12,14)
    ax.set_xticks([-10,-5,0,5,10])
    ax.set_xlabel(r'Spine position $[\mu m]$', fontsize=13)
    ax.set_ylim(-.2,.8)
    ax.set_ylabel(r'Relative change', fontsize=13, labelpad=1)

    # Right section
    for idx, (ax, t) in enumerate(zip(right_axs, [10,20,40])):
        ax.set_xlim(-12,14)
        ax.set_ylim(-.2,.8)
        ax.set_yticks([0.5])
        if idx == 0 or idx == 1: ax.set_xticklabels([])
        ax.text(16.5, 0.1, f'{t} min')

    ax = right_axs[-1]
    ax.set_xticks([-10,0,10])
    ax.set_xlabel(r'Spine position $[\mu m]$', fontsize=13)
    ax.text(-17.5, -0.1, 'Relative change', rotation=90, va='bottom', ha='right',
            fontsize=13)

    # Title
    # center_ax.set_title('Model prediction: 7 distributed stimulations protocol',
    #                     weight='semibold',
    #                     pad=30
    #                     )

    #
    # if save_folder:
    #     plt.savefig(join(save_folder, f'sim_7_distr.png'))
    #
    # plt.show()

    

