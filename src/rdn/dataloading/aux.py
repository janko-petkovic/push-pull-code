"""
Auxiliary methods for loading the data into dataframes
"""

import os
import pandas as pd
import numpy as np

def eggl_import(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    '''
    From path import Synapse_l.json and the background, returning them
    as a tuple

    Parameters
    ----------
    path : str
        Path to the directory to import from

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        The dataframe with the spine information and the array with the
        background information
    '''
    synapse_df = pd.read_json(os.path.join(path, 'dend_stat.json'))
    try:
        bg = np.load(os.path.join(path, 'backgroundM.npy')).squeeze()
        
    except:
        bg = np.load(os.path.join(path, 'background.npy')).squeeze()

    return synapse_df, bg



def eggl_transform(df:pd.DataFrame, bg: np.ndarray) -> pd.DataFrame:
    '''
    Given a synapse dataframe and a background numpy, build a new series with
    background subtracted RawIntDen 
    '''
    eggl_areas = []
    eggl_RID = df['RawIntDen'].map(lambda x: np.array(x, dtype='float'))

    for idx, spine in df.iterrows():
        eggl_bg = spine['area']*bg/(0.066**2)
        eggl_RID[idx] -= eggl_bg

    return eggl_RID
