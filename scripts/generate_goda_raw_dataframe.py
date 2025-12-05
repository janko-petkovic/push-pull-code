'''
Generates a pickled dataframe from the SypDen output regarding the Chater et al.
2024 GFP imaging dataset
'''

from rdn.dataloading import df_from_tool

if __name__ == '__main__':
    raw_df = df_from_tool(root='../data/TomData/',
                          dataset='goda',
                          compartment='spine')

    raw_df.to_pickle('../data/raw_data/raw_goda_data.pkl')
