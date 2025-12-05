import pandas as pd
import numpy as np

df = pd.read_csv('raw_data.csv', header=None)
 
Y = df[df[3]==' series'][1].to_numpy()
Y_err_low = df[df[3]==' err_low'][1].to_numpy()
Y_err = Y - Y_err_low
X = np.arange(Y.shape[0])
new_df = pd.DataFrame(np.stack((X, Y, Y_err), axis=0).T)
new_df.to_csv('data.csv', index=False, header=False)
