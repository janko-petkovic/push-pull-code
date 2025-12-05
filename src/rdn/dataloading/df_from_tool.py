"""
This function is a default for finding, testing and loading the data generated
with the Tool in a dataframe. Of course, you are free to rewrite this routine
with your custom needs
"""

import re
import os

import pandas as pd
import numpy as np
# from pandas.core.series import ValueKeyFunc
from scipy.stats import pearsonr

from . import create_path_pool
from .checkers import *
from .aux import eggl_import, eggl_transform


def df_from_tool(root: str,
                 dataset: str,
                 compartment: str) -> pd.DataFrame:
    """
    This is the final command that the user should be using to build a 
    dataframe from a set of jsons created with the tool.
    Theoretically, we should decide on a tree structure that the tool should
    use to organize its output but for now I use the variable dataset to
    switch between the two types of data that we have.
    The pipeline is as follows:
        1. create the path pool (all paths that contain all the target files)
        2. apply the path_checks (drop path if False)
        3. apply the dataframe_checks (drop row if False)

    Parameters
    ----------
    root : str
        path to the root folder from which to import the data (can be given
        relative to the call)

    dataset : str
        'goda' or 'helm', the dataset you are working with

    compartment: str
        'spine' or 'dendrite', the compartment you want to work with

    Returns
    -------
    pd.DataFrame
        a dataframe with all the information from the whole tree

    Notes
    -----
    I want to enforce the user to load everything and work with DataFrame
    views instead of loading pieces of the dataset.
    In the future, if our data gets to big, we could think of supporting
    partial loading (but I don't honestly think it is going to be too
    heavy to load as a single object).
    """
    
    # Pick the path pattern to use for the data extraction
    if dataset == 'goda': 
        pattern = r"CA1_(\d+(Spine|Distr))/(\w+)/cell_(\d+)"
    elif dataset == 'helm':
        pattern = r"(\w+)/merged/cell_(\d+)"
    else:
        raise ValueError(f'Invalid dataset specified: {dataset}')
        
    if compartment == 'spine':
        target_files = ['Synapse_l', 'ackground']

    elif compartment == 'dendrite':
        if dataset == 'goda': raise ValueError("Goda dataset has no dendritic compartment")
        target_files = ['dend_stat', 'ackground']
    else:
        raise ValueError(f'Invalid compartment: {compartment}')



    # Create the path pool
    # Files to be present in the directory paths

    # Check if the original and the background subtracted RID correlate
    # and exlude some uninteresting names
    exc_names = generate_check_names_not_in_path(['_MACOSX'])

    additional_checks = [exc_names]

    path_pool = create_path_pool(root = root,
                                 files_to_include = target_files,
                                 additional_checks = additional_checks)



    # Build the dataframe from the path_pool
    df_list = []

    # We will use this later for logging
    n_dropped_rows = 0
    n_dropped_cells = 0
    
    print("\n-> Applying additional dataframe checks...")


    #################################
    # MAIN LOOP 
    # Path by path.
    # If all the checks pass, we append the resulting dataframe to the
    # list of dataframes to concatenate in the end
    #################################
    for path in path_pool:
        
        # Import the data corpus
        # This is probably extremely stupid but whatever for now
        df = bg = None
        for target in target_files:
            for file in os.listdir(path):
                if target in file:
                    if 'json' in file and file[0] != '.':
                        df = pd.read_json(os.path.join(path, file))
                        if compartment == 'dendrite':
                            df = df.T
                    if 'M.npy' in file and file[0] != '.':
                        bg = np.load(os.path.join(path, file)).squeeze()

        if 0 in df.shape:
            print(path, ' empty .json!')
            continue 

        # BACKGROUND CHECK
        # See if we can subtract the background (cannot for now)
        try:
            if compartment == 'spine' and dataset == 'goda':
                # We really have to fix that background thing
                # df['egglRID'] = df['RawIntDen']
                df['egglRID'] = eggl_transform(df, bg)
            else:
                df['egglRID'] = df['RawIntDen']

        except Exception as err:
            print(path, err)
            continue


        # PARSING CHECK
        # Get the additional information depending on the dataset in use
        match = re.search(pattern, path)
        
        if dataset == 'goda':
            # Get number of stimulations, drug and cell informations
            try:
                df['nss'] = match[1]
                df['drug'] = match[3]
                df['cell'] = match[4]
            except:
                print(f"Failed matching at {path}")
                continue

        if dataset == 'helm':
            try:
                df['protein'] = match[1]
                df['cell'] = match[2]
            except:
                print(f"Failed matching at {path}")
                continue
                
                

        # GLOBAL DATAFRAME CHECKS

        # Check list consistency:
        # Same time point number
        if dataset == 'goda':
            
            # Check for consistent time points
            t_points = df['egglRID'].map(lambda x: len(x))
            
            if (t_points.nunique() != 1) or (t_points.iloc[0] != 8):
                print(f"Different time points in {path}, might want to check that out")
                continue
        
        # Same number of channels
        if dataset == 'helm':
            n_channels = df['egglRID'].map(lambda x: len(x))
            
            if (n_channels.nunique() != 1) or (n_channels.iloc[0] != 4):
                print(f"Different channels in {path}, might want to check that out")



        # CELL DATAFRAME CHECKS

        # Check if the number of stimulations is actually right. If not
        # drop the wrong cell
        if dataset == 'goda':

            # Get the number of stimulations
            nss = int(re.match('\d+', df.loc[0, 'nss'])[0])

            for cell in df['cell'].unique():
                cdf = df[df['cell']==cell]
                actual_nss = cdf[cdf['type'] == 'Stim'].shape[0]

                if abs(actual_nss - nss) > 1:
                    print(f'Inconsistent stims in {nss} - cell {cell} : {actual_nss}')
                    df = df[df['cell'] != cell]
                    n_dropped_cells += 1

        # Row by row checks (drop row if failed)
        # I dropped the checks on the EgglRID as i am not using it for now
        temp = []
        for idx, row in df.iterrows():

            # Check if there are negative or NaN values in egglRID
            # try:
            #     for val in row['egglRID']:
            #         if val < 0:
            #             raise Exception('Negative value')
            #         if pd.isna(val):
            #             raise Exception('NaN value')

            # except Exception as ex:
            #     print(f'   {path} | {ex} in egglRID at row {idx}. Dropping row.')
            #     n_dropped_rows += 1
            #     continue

            # # Check if egglRID correlates well with RawIntDen
            # r, p = pearsonr(row['RawIntDen'], row['egglRID'])
            # if r<0.9 or p>0.05:
            #     print(f"   {path} | bad correlation in row {idx} (r={r}, p={p}). Dropping row.")
            #     n_dropped_rows += 1
            #     continue

            temp.append(row)
            

            
        # Additional row checks ended, append whats left of the dataframe
        # to the final dataframe list
        if not temp:
            print(f"!  {path} | all rows have been dropped!")
            continue

        df = pd.DataFrame(temp)
            
            
        # Keep the ones with 8 time points anyway.
        # Moreover here we specify what stuff we are interested in
        if dataset == 'goda':
            # ATTENTION: dropped the use of egglRID because of the reduction in potentiation of stim spines
            df_list.append(df[['nss','drug','cell','type','distance','area','Times','RawIntDen','egglRID']][t_points==8])
            
            

        if dataset == 'helm':
            df_list.append(df[['protein','cell','egglRID','area']])
            
    print(f"   Dropped {n_dropped_cells} cells.")
    print(f"   Dropped {n_dropped_rows} rows.")

    final_df = pd.concat(df_list).reset_index(drop=True) 

    print(f"\n=> Final dataframe size: {final_df.shape}")
    return final_df

    
    
if __name__ == "__main__":
    
    df = df_from_tool("data/helm-spine-proteins", compartment='spine', dataset='helm')
    # df = df_from_tool("/home/janko/data-hangar/HeteroSynDataModel/TomData", dataset='goda', compartment='spine')

