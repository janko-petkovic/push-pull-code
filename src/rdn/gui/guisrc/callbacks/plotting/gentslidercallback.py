import numpy as np
from torch import arange, histogram, from_numpy

def _update_relative_errorbars(t,
                               avg_rsb,
                               sem_rsb,
                               error_obj,
                               error_obj_stim):

    # All the spines
    ln, (_), (bars,) = error_obj
    X = ln.get_xdata()
    Y, Y_err = avg_rsb[int(t)-2], sem_rsb[int(t)-2]
    
    yerr_top = Y+Y_err
    yerr_bot = Y-Y_err

    new_segments = [np.array([[x,yb], [x,yt]]) 
                    for x, yb, yt in zip(X, yerr_bot, yerr_top)] 
    ln.set_data(X,Y)

    bars.set_segments(new_segments)

    # Stimulated spines
    ln, (_), (bars,) = error_obj_stim
    X = ln.get_xdata()
    Y = avg_rsb[int(t)-2][X.astype(np.int8)]
    Y_err = sem_rsb[int(t)-2][X.astype(np.int8)]
    
    yerr_top = Y+Y_err
    yerr_bot = Y-Y_err

    new_segments = [np.array([[x,yb], [x,yt]]) 
                    for x, yb, yt in zip(X, yerr_bot, yerr_top)] 
    ln.set_data(X,Y)

    bars.set_segments(new_segments)



def _update_sizes_histogram(t, log_sb, hist_bins, bar_container):
    Y = log_sb[int(t)-2].flatten()
    n, _ = histogram(Y, from_numpy(hist_bins))

    for count, rect in zip(n, bar_container.patches):
        rect.set_height(count)


def _update_absolute_sizes(t, sb, sc_sb, sc_sb_stim):
    X = sc_sb.get_offsets().T[0]
    Y = sb[int(t)-2, :, 0]
    sc_sb.set_offsets([[x,y.item()] for x, y in zip(X, Y)])

    X = sc_sb_stim.get_offsets().T[0]
    Y = sb[int(t)-2, X, 0]
    sc_sb_stim.set_offsets([[x,y.item()] for x, y in zip(X, Y)])
    
    





# Slider Update function
def gen_t_slider_callback(avg_rsb, sem_rsb, error_obj, error_obj_stim, 
                          log_sb, hist_bins, bar_container,
                          sb, sc_sb, sc_sb_stim,
                          canvas):

    def t_slider_callback(t):
        # Update the errobars
        _update_relative_errorbars(t,
                                   avg_rsb,
                                   sem_rsb,
                                   error_obj,
                                   error_obj_stim)

        
        # Update the absolute sizes
        _update_absolute_sizes(t, sb, sc_sb, sc_sb_stim)
        # Update the histograms 
        _update_sizes_histogram(t, log_sb, hist_bins, bar_container)
        # Update some pointer on the time plot
        canvas.draw()

    return t_slider_callback

