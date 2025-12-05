'''
A function that creates the plots that are needed with the project with max
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def max_panel(p_ins):
    diffs = p_ins[1:] - p_ins[:-1]
    p_ins = p_ins[:-1]

    p_in = p_ins[:,0]

    flat_diffs = diffs.flatten()
    flat_p_ins = p_ins.flatten()

    df = pd.DataFrame({k:v for k, v in zip(flat_p_ins, flat_diffs)},
                      index=['var']).T.reset_index()
    df.rename(columns={'index':'size'},inplace=True)

    bins = pd.cut(df['size'], bins=6)
    gdf = df.groupby(bins)
    it = iter(gdf)

    s, i, r, p, _ = stats.linregress(flat_p_ins, flat_diffs)

    fig, axs = plt.subplots(1,2, figsize=(5,2))
    x = np.linspace(p_ins.min(), p_ins.max(), 100)

    ax = axs[0]
    ax.scatter(flat_p_ins, flat_diffs, s=2, color='gray', marker='x')
    ax.plot(x, s*x+i, c='tab:orange')
    ax.set_title(f'Instant size vs variation: r={r:.2}, p={p:.3}')

    ax = axs[1]
    ax.scatter(p_in[:-80]-p_in.mean(), p_in[80:]-p_in.mean())
    #next(it)[1]['var'].hist(bins=100, ax=ax, histtype='step')
    #next(it)[1]['var'].hist(bins=100, ax=ax, histtype='step')
    #next(it)[1]['var'].hist(bins=100, ax=ax, histtype='step')

    # ax.scatter(mean_spine_sizes, std_spine_vars, s=2, color='gray', marker='x')
    # ax.plot(x, ss*x+ii, c='tab:orange')
    # ax.set_title(f'mean size vs variation std: r={rr:.2}, p={pp:.3}')

