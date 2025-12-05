'''
Generates a pickled dataframe from the SypDen output regarding the Helm et al.
2022 multiplexed imaging.
'''

import os
import pandas as pd
from rdn.dataloading import df_from_tool

if __name__ == '__main__':

    df_raw = df_from_tool(root='../data/helm-spine-proteins/',
                          dataset='helm',
                          compartment='spine')

    # Split the channels separately
    channels = ['DoI', 'Homer', 'poi', 'poiHR']
    for i, ch in enumerate(channels):
        df_raw[ch] = df_raw['egglRID'].map(lambda x: x[i])

    # Convert areas to floats (remove once that Jean
    # corrects this)
    df = df_raw.dropna()

    for idx, row in df.iterrows():
        try:
            df.loc[idx, 'area'] = row['area']
        except:
            df.loc[idx, 'area'] = float(row['area'][0])

    df['area'] = df['area'].astype(float)
    pdf = df

    # import the calcineurin (different data structure)
    root = '../data/helm-spine-proteins/Calcineurin/merged'
    
    calci_dict = {
        'protein' : [],
        'DoI' : [],
        'Homer' : [],
        'poi' : [],
        'poiHR' : [],
        'area' : [],
    }
    
    for cell in os.listdir(root):
        dio = pd.read_csv(
            os.path.join(root, cell, 'Spine/Synapse_l_Channel_0.csv')
            )['Timestep 1 (RawIntDen)'].tolist()
    
        hom = pd.read_csv(
            os.path.join(root, cell, 'Spine/Synapse_l_Channel_1.csv')
            )['Timestep 1 (RawIntDen)'].tolist()
            
        poi = pd.read_csv(
            os.path.join(root, cell, 'Spine/Synapse_l_Channel_2.csv')
            )['Timestep 1 (RawIntDen)'].tolist()
        
        poiHR = pd.read_csv(
            os.path.join(root, cell, 'Spine/Synapse_l_Channel_3.csv')
            )['Timestep 1 (RawIntDen)'].tolist()

        area = pd.read_csv(
            os.path.join(root, cell, 'Spine/Synapse_l_Channel_3.csv')
            )['area'].tolist()

        calci_dict['protein'] += ['Calcineurin'] * len(hom)
        calci_dict['DoI'] += dio
        calci_dict['Homer'] += hom
        calci_dict['poi'] += poi
        calci_dict['poiHR'] += poiHR
        calci_dict['area'] += area
    
    calcidf = pd.DataFrame.from_dict(calci_dict)
    pdf = pd.concat((pdf, calcidf))
    
    pdf.to_pickle('../data/raw_data/raw_helm_data.pkl')
