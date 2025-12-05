def default_goda_filter(df):
    '''
    Every time we load the Goda dataset, we apply the baseline corrections
    settled upon in analysis-goda.ipynb (filter_spine_df).
    Every time we decide on a new filtering configuration, we have to update
    this method as well.

    For now we drop:
        - 3 stim abherrant spine 
        - 5 stim cell 7
        - 7 stim cell 8, 9
        - 15 stim cell 3, 6, 13
        - 7 distr cell 10
    '''
    fdf = df.copy()

    # Applying normalization baseline
    fdf.drop(fdf[(fdf['nss']==-1) & (
        (fdf['cell']=='10')
    )].index, inplace=True)

    fdf.drop(fdf[(fdf['nss']==5) & (
        (fdf['cell']=='7')
    )].index, inplace=True)

    fdf.drop(fdf[(fdf['nss']==7) & (
        (fdf['cell']=='9')
        | (fdf['cell']=='8')
    )].index, inplace=True)

    fdf.drop(fdf[(fdf['nss']==15) & (
        (fdf['cell']=='3')
        | (fdf['cell'] == '6')
        | (fdf['cell'] == '13')
    )].index, inplace=True)

    # Drop the hyperbig 3 stim intracluster spine
    fdf.drop(fdf[fdf['nss']==3]['distance'].idxmin(), inplace=True)

    return fdf