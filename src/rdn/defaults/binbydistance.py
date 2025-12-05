import pandas as pd
import numpy as np
import warnings


def bin_by_distance(df: pd.DataFrame,
                    bin_key: str,
                    bin_width: float,
                    bin_stride: float,
                    stim_at_zero: True) -> pd.DataFrame: 
    '''Bin a t_series using distance.

    Given a dataframe with a distance column and a t_series column, return a 
    new dataframe with binned values using the size specified in bin_size.

    The values included in each bin are selected with a right inclusive criterion,
    min_dist < dist <= max_dist.
       
    Returns
    -------
        pd.DataFrame 
        A dataframe with 4 columns: idx, distance, t_series, t_errs

    Note
    ----
        If you are binning a series, they should be already numpy!
    '''

    '''
    Start with the stim bin, drop it
    Start back with the intra bins and forward for the extra bins (subroutine?)
    Glue everything together 
    Has to work for float strides and bin_sizes as well!
    Can I use directly groupby? No cos we need the rolling and the rolling seems to be
    only implemented for a fixed number of rows or the offsets or whatever
    '''

    binned_df = []

    # Create the stim bin
    if stim_at_zero:
        stim_bin_view = df[df['distance']==0][bin_key].to_numpy()
        stim_mean = stim_bin_view.mean(axis=0)
        stim_stderr = stim_bin_view.std(axis=0)/np.sqrt(len(stim_bin_view))

        binned_df.append({
            'distance' : 0,
            'mean' : stim_mean,
            'stderr' : stim_stderr
        })

        # Drop the stimulations if we used them
        df = df[df['distance'] != 0]

    # We need to know how much we are supposed to step
    max_dist = df['distance'].max()
    min_dist = df['distance'].min()


    # We use [) binning so that we can use this same shitshow also without stimulation
    # Bin positives (extracluster)
    left_edge = -bin_width/2 + bin_stride

    while left_edge < max_dist:

        # Define the bin position and width. We assume dendrite orientation ->
        right_edge = left_edge + bin_width

        # Get bin and compute quantities
        bin_view = df[
            (df['distance'] >= left_edge)
            & (df['distance'] >= 0) # Im not starting with left edge on 0
            & (df['distance'] < right_edge)][bin_key].to_numpy()
        
        if len(bin_view) == 0: raise RuntimeError(left_edge, right_edge, 'nothing in this bin!')
        
        if len(bin_view) < 2:
            msg = f'{left_edge}, {right_edge} there is a 1 point bin! Associating 10% error'
            warnings.warn(msg)
            bin_mean = bin_view.mean(axis=0)
            bin_stderr = bin_mean/10

        else:
            bin_mean = bin_view.mean(axis=0)
            bin_stderr = bin_view.std(axis=0)/np.sqrt(len(bin_view))

        binned_df.append({
            'distance' : (right_edge + max(left_edge,0))/2,
            'mean' : bin_mean,
            'stderr' : bin_stderr
        })

        # Finally, step right the left edge
        left_edge += bin_stride

    # Bin negatives (intracluster) if present
    if min_dist < 0:
        right_edge = bin_width/2 -bin_stride

        while right_edge > min_dist:

            left_edge = right_edge - bin_width

            bin_view = df[
                (df['distance'] >= left_edge)
                & (df['distance'] < 0) # im nost startin with right edge on 0
                & (df['distance'] < right_edge)][bin_key].to_numpy()

            if len(bin_view) == 0: raise RuntimeError(left_edge, right_edge, 'nothing in this bin!')
            
            if len(bin_view) < 2:
                msg = f'{left_edge}, {right_edge} there is a 1 point bin! Associating 10% error'
                warnings.warn(msg)
                bin_mean = bin_view.mean(axis=0)
                bin_stderr = bin_mean/10

            else:
                bin_mean = bin_view.mean(axis=0)
                bin_stderr = bin_view.std(axis=0)/np.sqrt(len(bin_view))

            binned_df.append({
                'distance' : (min(right_edge, 0) + left_edge)/2,
                'mean' : bin_mean,
                'stderr' : bin_stderr
            })

            # Step left the right edge
            right_edge -= bin_stride


    binned_df = pd.DataFrame(binned_df).sort_values('distance').reset_index(drop=True)
        
    return binned_df





if __name__ == '__main__':
    df = pd.DataFrame({'distance' : [i for i in range(-2,10)], 'series' : [np.array([j+1 for i in range(10)]) for j in range(-2, 10)]})

    # print(df)
    print(bin_by_distance(df, 'series', 2.2, 1.7, stim_at_zero=True))


