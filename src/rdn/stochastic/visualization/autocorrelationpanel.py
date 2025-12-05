'''
Panel fitting the size distibutions at the last timepoint
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from ..stats import multifit_distribution, check_lognormal 

def autocorrelation_panel(p_ins):

    raw_y= p_ins
    y = raw_y - raw_y.mean(axis=0)
    
    # for spine in p_ins.T:
    #     if spine.mean() > 1300:
    #         y.append(spine)

    # y = np.stack(y, axis=1)

    fig, axs = plt.subplots(1,7, figsize=(12,1.3), dpi=250)
    plt.subplots_adjust(wspace=0.5, left=0.05, right=0.95, bottom=0.2)
    taus = np.array([5,10,17,25,35,45])
    # taus = [1,2,3,4,5,6]


    corrs = []
    stars = []

    for ax, tau in zip(axs, taus):
        s, i, r, p, e = stats.linregress(y[:-tau].flatten(), y[tau:].flatten())
        corrs.append(r)

        # plot
        x_plot = np.linspace(y[:-tau].flatten().min(), y[:-tau].flatten().max(), 10)
        y_plot = x_plot*s + i

        if p<0.05:
            line_color = 'tab:orange'
            stars.append('*')
        else:
            line_color = 'gray'
            stars.append('')

        ax.scatter(y[:-tau].flatten(), y[tau:].flatten(), color='gray', marker='.', s=1)
        ax.plot(x_plot,y_plot, color=line_color, linewidth=1, label=f'r = {r:.3}')
        ax.set_title(r'$\tau$' + f' = {tau}')
        ax.set_xlabel(r'$P_i$ at t = 0')
        ax.set_ylabel(r'$P_i$ at t = '+f'{tau}')
        ax.legend()

    ax = axs[-1]
    ax.plot(taus, corrs, linewidth=1)
    ax.scatter(taus, corrs, s=3)

    for i, star in enumerate(stars):
        ax.text(taus[i], corrs[i]+0.1, s=star)

    ax.set_ylim(-0.6,0.6)
    ax.set_xlabel(r'Autocorrelation time steps $\tau$')
    ax.set_ylabel(r'Pearson R coefficient')
    

