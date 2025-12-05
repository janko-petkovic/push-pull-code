'''
Panel fitting the size distibutions at the last timepoint
'''

import matplotlib.pyplot as plt
# plt.style.use('rdn.plotstyles.presentation')
import numpy as np
from scipy import stats

def fit_distr_panel(p_ins):
    y = p_ins[-1]
    log_y = np.log(y)
    dist = stats.norm
    
    fitres = dist.fit(log_y)
    print(stats.anderson(log_y))
    
    fig, ax = plt.subplots(1,1, figsize=(4,4), dpi=100)
    fig.tight_layout(rect=[0.1,0.1,1,1,])

    _, bins, _ = ax.hist(log_y, density=True, bins=50, color='tab:blue',
                         label='simulated data')
    ax.plot(bins, dist.pdf(bins, *fitres), color='tab:orange', linewidth=3,
            label='normal fit')
    ax.legend(frameon=False, fontsize=10)
    ax.set_xlabel(r'log $x_{in}$')
    ax.set_ylabel(r'pdf')

