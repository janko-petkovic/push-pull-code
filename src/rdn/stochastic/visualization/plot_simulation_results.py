'''
Method used for plotting the results of the simulated dendrite.
Optionally you can also plot the fits.
'''

import matplotlib.pyplot as plt
import numpy as np

def plot_simulation_results(model, # the model used for the simulation,
                            p_outs_t: np.ndarray,
                            p_ins_t: np.ndarray) -> None:
    '''
    Note: at a certain point we will change the model with the parameter
    values in time. The idea is to make this whole method interactive
    actually.
    '''
    fig = plt.figure(figsize=(6,3), dpi=100)
    gs = plt.GridSpec(2,4)

    # For now we are interested in the initial and the final situation
    p_outs_0 = p_outs_t[0]
    p_ins_0 = p_ins_t[0]

    p_outs_f = p_outs_t[-1]
    p_ins_f = p_ins_t[-1]
    t_final = len(p_ins_t)

    # For better visualization
    filter_dim = 1
    filter = np.ones(filter_dim)/filter_dim
    max_p = max(p_outs_0.max(), p_ins_0.max(), p_outs_f.max(), p_ins_f.max())

    # We now proceed from the left to the right of the plot
    # First the parameters along the dendrite (high) and distributions (low)
    ax_high = fig.add_subplot(gs[0,0])
    ax_low = fig.add_subplot(gs[1,0])

    for par_name, par_distr in model.__dict__.items():
        ax_high.plot(np.convolve(par_distr,filter, mode='valid'), label=f'{par_name}')
        ax_low.hist(par_distr, density=True, histtype='step',
                    label=f'{par_name}')

    ax_high.set_title("Parameter along dendrite")
    ax_high.set_xlabel("Position [um]")
    ax_high.set_ylabel("Magnitude")
    ax_high.legend()

    ax_low.set_title("Parameter freq histogram")
    ax_low.set_xlabel("Magnitude")
    ax_low.set_ylabel("pdf")
    ax_low.legend()

    # p_outs along the dendrite (high) and distributions (low)
    ax_high = fig.add_subplot(gs[0,1])
    ax_low = fig.add_subplot(gs[1,1])

    ax_high.plot(np.convolve(p_outs_0,filter, mode='valid'), label=f't = 0')
    ax_high.plot(np.convolve(p_outs_f,filter, mode='valid'), label=f't = {t_final}')
    ax_low.hist(p_outs_0, density=False, histtype='step', bins=50, label=f't = 0')
    ax_low.hist(p_outs_f, density=False, histtype='step', bins=50, label=f't = {t_final}')

    
    ax_high.set_ylim(0,max_p)
    ax_high.set_title("p_out along dendrite")
    ax_high.set_xlabel("Position [um]")
    ax_high.set_ylabel("Counts")
    ax_high.legend()

    ax_low.set_title("p_out freq histogram")
    ax_low.set_xlabel("Counts")
    ax_low.set_ylabel("Frequency")
    ax_low.legend()
    
    # p_ins along the dendrite (high) and distributions (low)
    ax_high = fig.add_subplot(gs[0,2])
    ax_low = fig.add_subplot(gs[1,2])

    ax_high.plot(np.convolve(p_ins_0,filter, mode='valid'), label=f't = 0')
    ax_high.plot(np.convolve(p_ins_f,filter, mode='valid'), label=f't = {t_final}')
    ax_low.hist(p_ins_0, density=True, histtype='step', bins=50, label=f't = 0')
    _, bins, _ = ax_low.hist(p_ins_f, density=True, histtype='step', bins=50,
                             label=f't = {t_final}')

    ax_high.set_ylim(0,max_p)
    ax_high.set_title("p_in along dendrite")
    ax_high.set_xlabel("Position [um]")
    ax_high.set_ylabel("Counts")
    ax_high.legend()

    ax_low.ticklabel_format(axis='y', style='scientific', scilimits=[0,0])
    ax_low.set_title("p_in freq histogram")
    ax_low.set_xlabel(r"$x_{in}$")
    ax_low.set_ylabel("pdf")
    ax_low.legend()
    

    # The fluctuation correlations during the process
    ax_high = fig.add_subplot(gs[0,3])
    ax_low = fig.add_subplot(gs[1,3])

    diffs_t = p_ins_t[1:] - p_ins_t[:-1]
    
    mean_pins = p_ins_t.mean(axis=0)
    std_pins = p_ins_t.std(axis=0)

    for idx in range(0, len(diffs_t)-1):
        ax_high.scatter(diffs_t[idx], diffs_t[idx+1], 1, 'C0', marker='x')

    ax_low.scatter(mean_pins, std_pins, 1, 'C0', marker='x')


    ax_high.set_title("p_in prev-next variations")
    ax_high.set_xlabel(r"$\Delta_{\,t}$")
    ax_high.set_ylabel(r"$\Delta_{\,t+1}$")

    ax_low.set_title(r"p_in $\sigma$ vs $\mu$")
    ax_low.set_xlabel(r"$\mu$")
    ax_low.set_ylabel(r"$\sigma$")

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
