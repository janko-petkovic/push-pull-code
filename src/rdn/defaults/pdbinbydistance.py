import pandas as pd
import numpy as np


def _std_fn(pdsrs):
    '''Auxiliary function for aggregating std of the mean'''
    try: 
        temp_arr = np.stack(pdsrs.to_list())
        std = temp_arr.std(axis=0)
        arr = std / np.sqrt(len(temp_arr))
    except: 
        # Unsafe: allows for empty bins or single spine bins
        print('Empty bins detected')
        arr = pdsrs
    return arr


def _count_fn(pdsrs):
    '''Auxiliary function for aggretating bin counts'''
    
    try:
        cnt = len(pdsrs)

    except: 
        # Unsafe: allows for empty bins or single spine bins
        print('Empty bins detected')

    return cnt



def pd_bin_by_distance(dff: pd.DataFrame,
                       bin_key: str,
                       bin_width: float) -> pd.DataFrame: 
    '''
    Given a dataframe with the usual eggl format, extract a series binned by distance.
       
    Returns
    -------
        pd.DataFrame 

    Note
    ----
        If you are binning a series, they should be already numpy!
    '''




    # Take out the stimulated statistics (this is a bin per se)
    sarr = np.stack(dff[dff['type']=='Stim'][bin_key].to_list(), axis=0)
    s_mean = sarr.mean(axis=0)
    s_std = sarr.std(axis=0)/np.sqrt(len(sarr))
    s_cnt = sarr.shape[0]

    # Calculate the bins
    n_bins_left = int(abs(dff['distance'].min()) / bin_width) + 2
    bins_left = -np.array([bin_width*i for i in range(n_bins_left)])[::-1]

    n_bins_right = int(dff['distance'].max() / bin_width) + 1
    bins_right = np.array([bin_width*i for i in range(n_bins_right)])

    bins = np.concatenate([bins_left[:-1], bins_right])


    dff = dff[dff['type']=='Spine']
    dff['Y'] = dff[bin_key]

    dff['bin'] = pd.cut(dff['distance'], bins)

    binned_dff = dff.groupby('bin').agg({
        'distance' : 'mean', 
        'Y' : ['mean', _std_fn, _count_fn],
        })

    binned_dff.reset_index(drop=True, inplace=True)


    # Add the stimulated bin again
    binned_dff.loc[-1] = [0, s_mean, s_std, s_cnt]


    # There will be nans bc of empty bins
    binned_dff.dropna(inplace=True)

    # return binned_dff

    # workaround to use to put 10% of mean instead of 0 std
    final_df = []

    for i, row in binned_dff.iterrows():

        # Handle single element bins
        if row['Y', '_std_fn'][0] == 0:
            final_df.append({
                'distance' : row.loc['distance', 'mean'],
                'mean' : row.loc['Y','mean'],
                'stderr' : row.loc['Y', 'mean']/10,
                'count' : 1
            })

        else:
            final_df.append({
                'distance' : row.loc['distance', 'mean'],
                'mean' : row.loc['Y','mean'],
                'stderr' : row.loc['Y', '_std_fn'],
                'count' : row.loc['Y', '_count_fn']
            })

    final_df = pd.DataFrame(final_df)

    # Sort by distance
    final_df.sort_values('distance', inplace=True)


    return final_df
    




if __name__ == '__main__':

    # x_min = -2
    # x_max = 20
    # x_delta = 0.5

    # t_min = 

    xs = np.linspace(-2,10,25)
    ts = np.linspace(0,40,5)

    mesh, _ = np.meshgrid(ts,xs)

    srs = [np.random.rand(len(ts)) for i in range(len(xs))]

    nss = np.ones_like(xs)
    types = 'Spine' * len(xs)

    df = pd.DataFrame({'distance' : xs.tolist(), 
                       'series' : srs,
                       'nss' : nss,
                       'type' : 'Spine'
                       })

    df.iloc[4, 3] = 'Stim'

    print(df)
    # df.drop(5, axis=0, inplace=True)
    # print(df)

    print(pd_bin_by_distance(df, 'series', 2.2))


