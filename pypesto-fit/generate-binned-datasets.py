'''
This script generates the binned datasets from to conduct the model fitting on starting
from the data of Chater et al. 2024 (with the addition of the 5 stimulation protocol).
To run it, just call

python generate-binned-datasets.py 

from the pypesto-fit folder.

The binned datasets will be saved in the pypesto-fit/binned-data folder.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdn.dataloading import df_from_tool
from rdn.defaults import default_goda_transformation, default_goda_filter, pd_bin_by_distance, mean_is_distance


if __name__ == '__main__':
    
    df = pd.read_pickle('../data/raw_data/raw_goda_data.pkl')

    df = df[df['drug']=='Control']
    df = default_goda_transformation(df)
    df = default_goda_filter(df)

    nsss = [1,3,5,7]

    # Plotting for visual inspection
    fig, axs = plt.subplots(1,4, figsize=(10,2))
    fig.tight_layout(pad=2)

    for ax, nss in zip(axs, nsss):
        dff = df[df['nss'] == nss].copy()
        
        # We stride the rolling windows using the mean interspine distance
        # found in the given experiment
        isd = mean_is_distance(dff)
        binned_df = pd_bin_by_distance(dff, 'RID', isd)
    
        
        t = np.array(df['Times'].iloc[0])
        x = np.array(binned_df['distance'])

        data = np.stack(binned_df['mean'].to_list(), axis=0).T
        data_errs = np.stack(binned_df['stderr'].to_list(), axis=0).T
        counts = binned_df['count'].to_numpy()

        # Control plot
        # control_idx = 3
        # ax.errorbar(x,data[control_idx], yerr=data_errs[0])
        # ax.axhline(y=1, linestyle='--', linewidth=0.5)
        # ax.axvline(x=0, linestyle='--', linewidth=0.5)

        # ax.set_title(f'{nss} stim')
        # ax.legend()

        # ax.set_ylim(0,2.5)

        # Save the dataset for later fitting
        # np.savetxt(f'binned-data/{nss}Spine_t.txt', t)
        # np.savetxt(f'binned-data/{nss}Spine_x.txt', x)
        # np.savetxt(f'binned-data/{nss}Spine_counts.txt', counts)
        # np.savetxt(f'binned-data/{nss}Spine_data.txt', data)
        # np.savetxt(f'binned-data/{nss}Spine_data_errs.txt', data_errs)

    # plt.show()

    print('[SUCCESS] Generated binned datasets.')