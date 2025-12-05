import tkinter as tk
import torch
import numpy as np 
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

from ..callbacks import run_simulation
from ..callbacks.plotting import gen_t_slider_callback


class PlotWindow(tk.Frame):
    
    def __init__(self, parent, input_entries, *args, **kwargs):
        
        super().__init__(parent, *args, **kwargs)
        self.parent = parent

        ##############
        # Simulation #
        ##############

        bsb, sb, rsb = [srs[2:] for srs in run_simulation(input_entries)]

        # Stimulation indexes for for later
        start_stim_idx = int(input_entries[3].get())
        n_stims = int(input_entries[4].get())
        stride_stims = int(input_entries[5].get())
        stim_idxes = [start_stim_idx + i * stride_stims
                      for i in range(n_stims)]
        
        example_idx = 0

        ############
        # Plotting #
        ############

        fig = plt.figure(figsize=(5,4), dpi=200)
        gs = fig.add_gridspec(4,5)


        #### Chart 1: relative sizes at time ####
        # Set the initial data
        sem_rsb, avg_rsb = torch.std_mean(rsb, dim=2)
        X = torch.arange(len(avg_rsb[0]))
        Y = avg_rsb[0]
        Y_err = sem_rsb[0]

        # Errorbar
        # So, appearently, you cannot color the points of the errorbar
        # in different colors (you can do it only with the bars).
        # Therefore I have to plot a second time the stimulated spines.
        # If you find a different way please tell me.


        ax = fig.add_subplot(gs[:-2,:3])
        rerror_obj = ax.errorbar(X, Y, Y_err, fmt='.', 
                                 label='unstimulated', picker=True)

        rerror_obj_stim = ax.errorbar(X[stim_idxes], Y[stim_idxes], 
                                      Y_err[stim_idxes], fmt='.', 
                                      label='stimulated')


        ax.axhline(y=1, linestyle=':', linewidth=1, alpha=0.3, c='black',
                   label='baseline')

        # Set the axis limits
        y_lim_max = (avg_rsb + sem_rsb).max() * 1.1
        y_lim_min = (avg_rsb - sem_rsb).min() * 0.5
        y_lim_min = min(y_lim_min.item(), 0.5)
        ax.set_ylim(y_lim_min, y_lim_max)

        ax.set_xlabel('Spine position [um]')
        ax.set_ylabel('Normalized P')
        ax.legend(frameon=False, loc='upper center', ncols=3)


        #### Chart 2: absolute P at time ####
        # Set the initial data
        X = torch.arange(len(sb[0,:,example_idx]))
        Y = sb[0,:,example_idx]
        Y_base = bsb[0,:,example_idx]

        # Draw the figure
        ax = fig.add_subplot(gs[-2:,:-2])
        ax.scatter(X,Y_base, s=5, c='white', linewidths=0.5, edgecolors='black',
                   alpha=0.3, label='baseline')
        sc_sb = ax.scatter(X,Y, s=5, label='unstimulated')
        sc_sb_stim = ax.scatter(X[stim_idxes], Y[stim_idxes], s=5,
                                label='stimulated')

        # Set the axis limits
        y_lim_max = sb[:,:,example_idx].max() * 10
        y_lim_min = sb[:,:,example_idx].min() / 10
        ax.set_ylim(y_lim_min, y_lim_max)

        ax.set_yscale('log')
        ax.set_xlabel('Spine position on dendrite [um]')
        ax.set_ylabel('P')
        ax.legend(frameon=False, 
                  loc='upper center',
                  ncol=3)



        #### Chart 3: selected spine relative in time
        sem_rsb, avg_rsb = torch.std_mean(rsb, dim=2)
        Y = avg_rsb.T[0]
        Y_err = sem_rsb.T[0]
        X = torch.arange(len(Y))

        ax = fig.add_subplot(gs[:2,3:])
        ax.errorbar(X, Y, Y_err, fmt='.')

        y_lim_max = (avg_rsb + sem_rsb).max() * 1.1
        y_lim_min = (avg_rsb - sem_rsb).min() * 0.5
        y_lim_min = min(y_lim_min.item(), 0.5)

        # ax.set_ylim(y_lim_min, y_lim_max)

        ax.set_xlabel('Time after stimulation [min]')
        ax.set_ylabel('Normalized P (0 um)')


        #### Chart 4: P histogram at time ####
        ax = fig.add_subplot(gs[2:,3:])

        log_sb = sb.log()
        Y = log_sb[0].flatten()
        Y_base = bsb[0].flatten().log()

        hist_bins = np.linspace(log_sb.min(),
                                log_sb.max(),
                                30)

        _, _, bar_container = ax.hist(Y, hist_bins, label='current')
        ax.hist(Y_base, hist_bins, histtype='step', alpha=0.3, color='black',
                label='baseline')

        x_lim_min = min(Y_base.min(), Y_base.min())*0.9
        x_lim_max = max(Y.max(), Y_base.max())*1.1
        ax.set_xlim(x_lim_min, x_lim_max)

        ax.set_xlabel('Log P')
        ax.set_ylabel('Counts')
        ax.legend(frameon=False)



        #### Common parts and sliders
        # Canvas 
        def onpick(event):
            if isinstance(event.artist, plt.Line2D):
                print(event.artist.get_xdata()[event.ind])
            # print(event.artist.get_path())

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.mpl_connect('pick_event', onpick)
        canvas.draw()

        # Time slider 
        t_slider_callback = gen_t_slider_callback(# Relative sizes at time
                                                  avg_rsb,
                                                  sem_rsb,
                                                  rerror_obj,
                                                  rerror_obj_stim,

                                                  # Histogram
                                                  log_sb,
                                                  hist_bins,
                                                  bar_container,

                                                  # Example absolute sizes at time
                                                  sb,
                                                  sc_sb,
                                                  sc_sb_stim,
                                                  canvas)

        slider_update = tk.Scale(master=self,
                                 from_=2, 
                                 to=int(input_entries[0].get()),
                                 orient=tk.HORIZONTAL,
                                 command=t_slider_callback,
                                 label='Time [min]')
        
        # Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(canvas=canvas,
                                       window=self,
                                       pack_toolbar=False)

        # Packing (from bottom up cos the guy said things about resizing and
        # stuff - dunno, imma just trust him)
        slider_update.pack(side=tk.BOTTOM)
        toolbar.pack(side=tk.BOTTOM)
        plt.subplots_adjust(wspace=2, hspace=2)
        canvas.get_tk_widget().pack(side=tk.BOTTOM)

