import numpy as np


def default_goda_transformation(in_df):
    '''
    The transformations we always need when dealing with the Goda df
        - use ints to refer to nss (number of stimulations) - 7 distributed
          goes to -1
        - numpy array the luminosity signal
        - create a base RID (mean of prestim sizes)
        - create luminosity columns for each time point
        - create ratio and delta values for each time point
    '''
    df = in_df.copy()

    # Substitute the nss names with the number of stimulations
    df['nss'].replace({
        '15Spine' : 15,
        '1Spine' : 1,
        '3Spine' : 3,
        '5Spine' : 5,
        '7Spine' : 7,
        '7Distr' : -1
    }, inplace=True)

    # Sort by number of stimulations
    df.sort_values('nss', inplace=True)

    # Cast RID to array
    df['RID'] = df['RawIntDen'].map(lambda x: np.array(x))
    
    # Basal RID as mean pre-stimulus
    df['base_RID'] = df['RID'].map(lambda x: x[:3].mean())
    
    # Normalize RID to basal RID
    df['norm_RID'] = df['RID'].map(lambda x: x/x[:3].mean())
    
    # Variation of RID compared to basal RID
    df['delta_RID'] = df['RID'].map(lambda x: x - x[:3].mean())

    # Create single point values from time series (maybe needed)
    ts = [-15,-10,-5,2,10,20,30,40]

    for i, t in enumerate(ts):
        df[f'{t}'] = df['RID'].map(lambda x: x[i])
        df[f'norm_{t}'] = df['norm_RID'].map(lambda x: x[i])
        df[f'delta_{t}'] = df['delta_RID'].map(lambda x: x[i])

        if t != -15:
            df[f'step_{t}'] = df[f'{t}'] - df[f'{ts[i-1]}']

    return df

    