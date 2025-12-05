"""
This function is a default for finding, testing and loading the data generated
with the Tool in a dataframe. Of course, you are free to rewrite this routine
with your custom needs
"""

import re

import pandas as pd

from tqdm import tqdm

from ..dataloading import create_path_pool
from ..dataloading.checkers import *
from ..dataloading.aux import eggl_import, eggl_transform


def df_from_tool(root: str,
                 dataset: str,
                 drugs: list[str] = []) -> pd.DataFrame:
    
    # Pick the path pattern to use for the data extraction
    if dataset == 'max':
        pattern = r"CA1_(\d+(Spine|Distr))/(\w+)/cell_(\d+)"
        
    elif dataset == 'kanaan':
        pattern = r"(\w+)/merged/cell_(\d+)"
        
    else:
        raise ValueError(f'Invalid dataset specified: {dataset}')
        


    # Create the path pool
    
    # Files to be present in the directory paths
    target_files = ['Synapse_l', 'ackground']

    # Check if the original and the background subtracted RID correlate
    # and exlude some uninteresting names
    exc_names = generate_check_names_not_in_path(['_MACOSX'])
    rid_erid_corr_check = generate_check_t_series_corr(col_1='RawIntDen',
                                                       col_2='egglRID',
                                                       corr_type='pearson',
                                                       min_r=0.9,
                                                       max_p=0.05)

    additional_checks = [exc_names, rid_erid_corr_check]

    if drugs:
        inc_names = generate_check_names_in_path(drugs)
        additional_checks.insert(0, inc_names)

    path_pool = create_path_pool(root = root,
                                 files_to_include = target_files,
                                 additional_checks = additional_checks)

    # Build the dataframe from the path_pool
    df_list = []
    
    

    for path in tqdm(path_pool):
        
        # Import the data corpus
        df, bg = eggl_import(path)
        df['egglRID'] = eggl_transform(df, bg)

        # Get the additional information depending on the dataset in use
        match = re.search(pattern, path)
        
        if dataset == 'max':
            # Get number of stimulations, drug and cell informations
            try:
                df['nss'] = match[1]
                df['drug'] = match[3]
                df['cell'] = match[4]
            except:
                print(f"Failed matching at {path}")

        if dataset == 'kanaan':
            try:
                df['protein'] = match[1]
                df['cell'] = match[2]
            except:
                print(f"Failed matching at {path}")
                
                
        # Check for negative values in the egglRID series (the only that could
        # have them theoretically)
        temp = []
        for idx, row in df.iterrows():
            try:
                for val in row['egglRID']:
                    if val < 0:
                        raise Exception('Negative value')
                    if pd.isna(val):
                        raise Exception('NaN value')
            except Exception as ex:
                print(f'{path} | {ex} in row {idx}, dropping row.\n')
                continue
            temp.append(row)
        df = pd.DataFrame(temp)


        # Check for time serieses with different length
        if dataset == 'max':
            #t_points_flag = False
            t_points = df['egglRID'].map(lambda x: len(x))
            
            if (t_points.nunique() != 1) or (t_points.iloc[0] != 8):
                #t_points_flag = True
                print(f"Different time points in {path}, might want to check that out")
        
        # Check that all four channels are present
        if dataset == 'kanaan':
            #channel_flag = False
            n_channels = df['egglRID'].map(lambda x: len(x))
            
            if (n_channels.nunique() != 1) or (n_channels.iloc[0] != 4):
                #channel_flag = True
                print(f"Different channels in {path}, might want to check that out")
            
            
        # Keep the ones with 8 time points anyway.
        # Moreover here we specify what stuff we are interested in
        if dataset == 'max':
            df_list.append(df[['nss','drug','cell','type','distance','area','Times','egglRID']][t_points==8])
            
        if dataset == 'kanaan':
            df_list.append(df[['protein','cell','egglRID']])
            

    return pd.concat(df_list).reset_index(drop=True) 

    
    

