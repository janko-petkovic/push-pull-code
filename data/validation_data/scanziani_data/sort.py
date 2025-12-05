import pandas as pd
import numpy as np

df = pd.read_csv('raw_microcystine.csv', header=None)
 
X = df[df[3]==' series'][0].to_numpy()
Y = df[df[3]==' series'][1].to_numpy()
Y_err_high = df[df[3]==' err_up'][1].to_numpy()
Y_err = Y_err_high - Y
new_df = pd.DataFrame(np.stack((X, Y, Y_err), axis=0).T)
new_df.to_csv('microcystine.csv', index=False, header=False)
