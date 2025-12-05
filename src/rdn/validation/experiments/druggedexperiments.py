from math import log
from os.path import join
from copy import deepcopy

import torch 
torch.manual_seed(2)

import numpy as np
np.random.seed(2)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import mannwhitneyu

from ...defaults import (default_goda_transformation,
                         stripplot)

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


def plot_drugged_section(axs, df, model, model_p_dict, simulation_time,
                         spine_number, inter_spine_distance):

    t_idxes = [2, 20, 30, 40]
    n_experiments = 100
    stim_configurations = [torch.arange(110,390,40)]

    # Range we take for the visualization
    lr = 11
    ur = 15

    # Stimulated index for the visualization
    stim_idx = 350


    rel_sizes_batch = None

    for stim_idxes in stim_configurations:
        simulation = Simulation(model = model,
                                model_p_dict = model_p_dict,
                                simulation_time = simulation_time,
                                spine_number = spine_number,
                                inter_spine_distance = inter_spine_distance,
                                stim_indexes = stim_idxes)

        _, _, rel_sizes_batch = simulation.run(n_experiments)

    center_ax = axs[2]
    plot_prediction_plot(center_ax, df, rel_sizes_batch, bins, lr, ur, stim_idx,
                         2, linewidth=3, markersize=10, capsize=2, s=10)


def drugged_experiments(model: BaseModel,
                            model_p_dict: dict,
                            simulation_time: int = 40, 
                            spine_number: int = 200,
                            inter_spine_distance: int = 1,
                            bins: list = [-15,-10,-5,0,5,10,15],
                            axs: list = [], save_folder: str = '') -> None:

    ##############
    # PARAMETERS #
    ##############

    model_p_dict['Pi'] /= 13
    n_experiments = 10
    stim_configurations = [torch.arange(110,390,40)]
    stim_idxes = torch.tensor([100,102,105,107,110,112,115])
    
    # stupidest possible way but whatever for now
    cl_idxes = torch.tensor([98,99,101,103,104,106,108,109,111,113,114,116,117])


    # Range we take for the visualization
    lr = 11
    ur = 15

    # Stimulated index for the visualization
    stim_idx = 350

    nss = 7
    time_col = 'norm_10'
    time_index = 10

    ################
    # DATA LOADING #
    ################

    # Import the experimental dataset
    df = df_from_tool(
            root='data/TomData',
            dataset='goda',
            compartment='spine'
        )

    df = default_goda_transformation(df)

    # Select the drugs
    cdf = df[df['drug'] == 'Control']
    aipdf = df[df['drug'] == 'CamKII']
    fkdf = df[df['drug'] == 'Calcineurin']


    # Baseline
    Yc_base = cdf['RID'].map(lambda x: x[0])
    Yaip_base = aipdf['RID'].map(lambda x: x[0])
    Yfk_base = fkdf['RID'].map(lambda x: x[0])

    # Stimulated spines
    Yc_stim = cdf[(cdf['nss']==nss) & (cdf['type']=='Stim')][time_col]
    Yaip_stim = aipdf[(aipdf['nss']==nss) & (aipdf['type']=='Stim')][time_col]
    Yfk_stim = fkdf[(fkdf['nss']==nss) & (fkdf['type']=='Stim')][time_col]

    # Cluster spines
    Yc_cl = cdf[(cdf['nss']==nss) & (cdf['distance']<0)][time_col]
    Yaip_cl = aipdf[(aipdf['nss']==nss) & (aipdf['distance']<0)][time_col]
    Yfk_cl = fkdf[(fkdf['nss']==nss) & (fkdf['distance']<0)][time_col]


    ##############
    # SIMULATION #
    ##############

    # Control run
    simulation = Simulation(model = model,
                            model_p_dict = model_p_dict,
                            simulation_time = simulation_time,
                            spine_number = spine_number,
                            inter_spine_distance = inter_spine_distance,
                            stim_indexes = stim_idxes)

    bsb, _, rsb = simulation.run(n_experiments)

    SYc_base = bsb[time_index,:,:].flatten()
    SYc_stim = rsb[time_index,stim_idxes,:].flatten()
    SYc_cl = rsb[time_index,cl_idxes,:].flatten()


    # # AIP run
    # aip_model_p_dict = deepcopy(model_p_dict)
    #
    # aip_model_p_dict['Ks'] /= 5
    # aip_model_p_dict['mu_log_K_N'][0] -= log(1.5)
    #
    # simulation = Simulation(model = model,
    #                         model_p_dict = aip_model_p_dict,
    #                         simulation_time = simulation_time,
    #                         spine_number = spine_number,
    #                         inter_spine_distance = inter_spine_distance,
    #                         stim_indexes = stim_idxes)
    #
    # bsb, _, rsb = simulation.run(n_experiments)
    #
    # SYaip_base = bsb[time_index,:,:].flatten()
    # SYaip_stim = rsb[time_index,stim_idxes,:].flatten()
    # SYaip_cl = rsb[time_index,cl_idxes,:].flatten()


    # fk506 run
    fk_model_p_dict = deepcopy(model_p_dict)
    fk_model_p_dict['Ns'] /= 1.8

    fk_model_p_dict['mu_log_K_N'][1] -= log(1.5)
    fk_model_p_dict['sigma_N'] /= 1.3


    simulation = Simulation(model = model,
                            model_p_dict = fk_model_p_dict,
                            simulation_time = simulation_time,
                            spine_number = spine_number,
                            inter_spine_distance = inter_spine_distance,
                            stim_indexes = stim_idxes)

    bsb, _, rsb = simulation.run(n_experiments)

    SYfk_base = bsb[time_index,:,:].flatten()
    SYfk_stim = rsb[time_index,stim_idxes,:].flatten()
    SYfk_cl = rsb[time_index,cl_idxes,:].flatten()
    
    from scipy.stats import kstest
    def combo_plot(ax, Ys, SYs, **kwargs):
        Yc, Yfk = Ys
        SYc, SYfk = SYs

        print(mannwhitneyu(Yc, Yfk, alternative='less'))

        stripplot(ax,[Yc, Yfk], s=kwargs.get('s'), alpha=0.1, linewidths=0,
                  c='gray', jitter=0.3) 

        ax.boxplot([Yc, Yfk],
                   showfliers=False,
                   boxprops=dict(linewidth=1),
                   medianprops=dict(color='black', linewidth=1),
                   whiskerprops=dict(linewidth=1),
                   capprops=dict(linewidth=1),
                   )

        medians = (
                SYc.median(),
                SYfk.median(),
                )
        q1 = (
                SYc.quantile(0.25),
                SYfk.quantile(0.25),
        )

        q3 = (
                SYc.quantile(0.75),
                SYfk.quantile(0.75),
        )
        ax.plot((1,2),medians,linewidth=3)
        ax.fill_between((1,2),q1, q3, alpha=0.2)


    fig, axs = plt.subplots(1,3, figsize=(12,3))

    combo_plot(axs[0],
               [Yc_base, Yfk_base],
               [SYc_base, SYfk_base],
               s=10,
               )

    combo_plot(axs[1],
               [Yc_stim, Yfk_stim],
               [SYc_stim, SYfk_stim],
               s=20,
               )

    combo_plot(axs[2],
               [Yc_cl, Yfk_cl],
               [SYc_cl, SYfk_cl],
               s=20,
               )

    # ax.set_xticklabels(['Control', 'AIP', 'FK506'], rotation=45)
    # ax.set_ylabel('Fluorescence [a.u.]')


