'''
Method used for plotting the results of the simulated dendrite.
Optionally you can also plot the fits.
'''

import matplotlib.pyplot as plt
# plt.style.use('rdn.plotstyles.rdnstyle')
import numpy as np
from scipy import odr
from scipy.stats import shapiro, norm

def paper_figure(model, # the model used for the simulation,
                 p_outs_t: np.ndarray,
                 p_ins_t: np.ndarray) -> None:
    '''
    We are trying to reproduce fig. 5 from Yasumatsu 2008 and we can also
    add Loevenstein 2011 fig. 4
    '''
    # For now we are interested in the initial and the final situation
    p_ins_0 = p_ins_t[0]
    p_ins_f = p_ins_t[-1]
    
    mean_p_ins = p_ins_t.mean(axis=0)
    std_p_ins = p_ins_t.std(axis=0)

    diffs_t = p_ins_t[1:] - p_ins_t[:-1]
    mean_diffs = diffs_t.mean(axis=0)
    std_diffs = diffs_t.std(axis=0)
  
    # Fit the logarithms and test for normality
    log_y = np.log(p_ins_f)
    shapres = shapiro(log_y)
    normres = norm.fit(log_y)
    print(shapres.pvalue)
    normcolor = 'tab:orange' if shapres.pvalue > 0.05 else 'gray'
     

    # Bin the diffs for better visualization (right)
    bins = np.linspace(mean_p_ins.min(), mean_p_ins.max()+10, 6) 
    inds = np.digitize(mean_p_ins, bins)
    bmean_p_ins = []
    bstd_p_ins = []
    bmean_diffs = []
    bstd_diffs = []
    
    # BINNING
    for i in set(np.sort(inds)):
        idxs = np.where(inds == i)
        bmean_p_ins.append([np.mean(mean_p_ins[idxs]),
                           np.std(mean_p_ins[idxs])])
        bstd_p_ins.append([np.mean(std_p_ins[idxs]),
                           np.std(std_p_ins[idxs])])
        bmean_diffs.append([np.mean(mean_diffs[idxs]),
                            np.std(mean_diffs[idxs])])
        bstd_diffs.append([np.std(std_diffs[idxs]),
                            np.std(std_diffs[idxs])])

    bmean_p_ins = np.stack(bmean_p_ins, axis=0).T
    bstd_p_ins = np.stack(bstd_p_ins, axis=0).T
    bmean_diffs = np.stack(bmean_diffs, axis=0).T
    bstd_diffs = np.stack(bstd_diffs, axis=0).T

    # Fit the variance data to the basal sizes
    def f(p, x):
        return p[0] + p[1]*x
    linear = odr.Model(f)
    data = odr.RealData(bmean_p_ins[0], bstd_p_ins[0],
                    sx=bmean_p_ins[1], sy=bstd_p_ins[1])
    problem = odr.ODR(data, linear, beta0=[1.,1e-3])
    out = problem.run()
    print(out.beta + out.sd_beta)


    ##########
    # Plotting
    ##########
    rdngreen = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    fig = plt.figure(figsize=(8,3), dpi=100)
    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.98,top=0.9,
                        wspace=0.3)
    gs = fig.add_gridspec(2,2)
    
    x_line = np.array([2e3, bmean_p_ins[0].max()*1.1])
   
    # Sizes histogram
    ax = fig.add_subplot(gs[:,0])
    _, _, patches = ax.hist(p_ins_f, density=True, label='Simulated data')
    ax.set_xlabel(r'$y_i$ at fixed time [a.u.]')
    ax.set_ylabel('pdf')

    axi = fig.add_axes([0.28, 0.7, 0.2,0.2])
    _, bins, _ = axi.hist(np.log(p_ins_0), density=True)
    line = axi.plot(bins, norm.pdf(bins, *normres), c=normcolor)
    axi.set_xticks([])
    # axi.set_yticks([])
    axi.set_xlabel('Log $y_i$', size=10)
    
    ax.legend([patches, line[0]], 
              ['Simulated data', 
               f'Normal fit\nShapiro p-value = {shapres.pvalue:.2}'], 
              loc=[0.5,0.2], fontsize=8)

    # Mean variation
    ax = fig.add_subplot(gs[0,1])
    ax.errorbar(bmean_p_ins[0], bmean_diffs[0], 
                yerr=bmean_diffs[1], 
                xerr=bmean_p_ins[1], 
                fmt='.', markersize=9, elinewidth=1.5,
                label='Simulated data')
    ax.axhline(y=0, linewidth=1, linestyle=(0,(5,5)),
               c='black')

    ax.set_xlim(x_line.min(), x_line.max())
    ax.set_xticklabels('')
    ax.set_ylabel('$\mu_{\Delta y_i}}$ [a.u.]')
    ax.legend(fontsize=8, loc=[0.02, 0.8])
  
    
    # Variation vs size
    ax = fig.add_subplot(gs[1,1])
    ax.scatter(p_ins_0, std_p_ins, s=1, alpha=0.2)
    # ax.errorbar(bmean_p_ins[0], bstd_diffs[0], 
    #             yerr=bstd_diffs[1],
    #             xerr=bmean_p_ins[1], 
    #             fmt='.', markersize=9, elinewidth=1.5, 
    #             zorder=0, label='Simulated data')
    # ax.errorbar(bmean_p_ins[0], bstd_p_ins[0], 
    #             yerr=bstd_p_ins[1],
    #             xerr=bmean_p_ins[1], 
    #             fmt='.', markersize=9, elinewidth=1.5, 
    #             zorder=0, label='Simulated data')
    # ax.plot(x_line, f(out.beta, x_line), zorder=1,
    #         c=rdngreen, label='Linear fit')
    #
    # ax.fill_between(x_line,
    #                 f(out.beta-out.sd_beta, x_line),
    #                 f(out.beta+out.sd_beta, x_line),
    #                 alpha=0.1)
    # ax.set_xlim(x_line.min(), x_line.max())
    # ax.set_xlabel('Time averaged $y_{i}$ [a.u.]')
    # ax.set_ylabel('$\sigma_{y_i}$ [a.u.]')
    # ax.legend(loc=[0.02,0.6], fontsize=8)

    # ax.plot(p_ins_t.T[np.argmax(mean_p_ins)])
    # ax.plot(p_ins_t.T[np.argmin(mean_p_ins)])
    # ax.set_yscale('log')

    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
