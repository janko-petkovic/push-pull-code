import numpy as np


def mean_is_distance(dff):
    '''
    Given a dataframe with spines find the mean interspine
    observed among the different cells.
    
    Caveat
    ------
    It will calculate the mean across ALL cells, so keep in mind that 
    you have to select your samples before calling the function.

    Also, given how this algorythm works, one can only use the spines lying
    outside of the stimulation interval.
    '''
    cells = dff['cell'].unique()

    distss = []

    # Take interspine distance for every cell
    # Notice how we are sampling only half of the spines!
    for cell in cells:
        dffc = dff[(dff['cell'] == cell) & (dff['distance']>0)].sort_values('distance')

        # dffc contains spines from both the left and the right outside-of-stimulation
        # regions (double the density). Our best guess is to take only half of the spines
        # (Wilcoxon thing whatever)
        dffc = dffc.iloc[::2]
        
        x = dffc['distance'].sort_values().to_numpy()
        dists = x[1:] - x[:-1]

        distss.append(dists)

    # The fit is here
    return np.concatenate(distss).mean()