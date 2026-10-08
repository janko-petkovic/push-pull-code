import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt

    from scipy.optimize import least_squares

    from rdn.validation import Simulation
    from rdn.defaults import pardict_from_result
    # from rdn.fitting.models import LocalGaussModelTilde
    from rdn.rebuttal import scipymodels

    return (
        Path,
        least_squares,
        mo,
        np,
        pardict_from_result,
        pd,
        plt,
        scipymodels,
    )


@app.cell
def _(Path, plt):
    plt.style.use("default")
    plt.rcParams["font.family"] = "open sans"

    ROOT = Path(__file__).parent
    return (ROOT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Set up the model parameters
    """)
    return


@app.cell
def _(LocalGaussModelTilde, ROOT, pardict_from_result):
    _model = LocalGaussModelTilde()

    model_p_dict = pardict_from_result(
        ROOT / (
            "output/multi_fitting/Multi_LocalGaussModelTilde/"
            "NLLAdast/1_3_5_7_Spine_data_fides_1200.hdf5"
        ),
        Chi=1,
        dendrite_length=1000,
        N_mean=5000,
        run_index=0,
    )

    simulation_time = 40
    spine_number = 200
    inter_spine_distance = 1
    model_p_dict["Pi"] = model_p_dict["Pi"] / 10
    model_p_dict["tau_N"] = model_p_dict["tau_N"] * 2

    # Usual accounting for shorter dendrite
    # As discussed
    model_p_dict["tau_K"] = model_p_dict["tau_K"] * 2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Import the datasets
    """)
    return


@app.cell
def _(np, pd):
    dflist = []
    dflist.append(pd.read_csv("data/validation_data/chindemi_data/banerjee.csv").rename(columns={'x':'pre', ' y':'gamma'}))
    dflist.append(pd.read_csv("data/validation_data/chindemi_data/egger.csv").rename(columns={'x':'pre', ' y':'gamma'}))
    dflist.append(pd.read_csv("data/validation_data/goda_data/stim_norm_2_vs_base.csv").rename(columns={'base_RID':'pre', 'norm_2':'gamma'}))

    for i, df in enumerate(dflist):
        dflist[i]['log_gamma'] = np.log(df['gamma'])
        dflist[i]['post'] = df['pre'] * df['gamma']

        log_pre = np.log(df['pre'])
        dflist[i]['stz_pre'] = np.exp((log_pre - log_pre.mean())/log_pre.std())


    index_list = []
    for _dff, name in zip(dflist, ('bdf', 'edf', 'gdf')):
        index_list += [(name, i) for i in _dff.index]

    indexes = pd.MultiIndex.from_tuples(index_list, names=['set', 'idx'])

    def standardize_pre(df):
        log_pre = np.log(df['pre'])

        df['pre'] = np.exp((log_pre - log_pre.mean())/log_pre.std())
        # df['pre'] = df['pre'] / df['pre'].max()
        return df


    df = pd.DataFrame(
        pd.concat(standardize_pre(d) for d in dflist),
    ).reset_index(drop=True).set_index(indexes)

    df.loc[['bdf', 'gdf'], 'gamma']
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From the rank estimation we have, for 38 points bins, that, in ranks:

    | x | low | est | high |
    |-|-|-|-|
    | Q1 | 5 | 10 | 15 |
    | med | 14 | 19 | 25 |
    | Q3 | 24 | 29 | 34 |
    """)
    return


@app.cell
def _(df, least_squares, scipymodels):
    model_dict = {str(m) : m for m in [
        scipymodels.PowerLaw(),
        scipymodels.PowerLawScale(),
        scipymodels.PushPullMedian(),
    ]}

    X =  df.loc[['edf','gdf']]['stz_pre']
    Y = df.loc[['edf','gdf']]['log_gamma']
    res = {}

    for _modelname, _model in model_dict.items():
        res[_modelname] = least_squares(
            _model.residual,
            x0=_model.gen_p0(),
            args=(X,Y)
        )

    for k,v in res.items():
        print(k, v.x)
    # print('RDN')
    # print('---------')
    # for k, v in res_rdn.items():
    #     print(f"{k:20}", '\t', 10**v.x)

    # print('')
    # print('Power law')
    # print('---------')
    # for k, v in res_power.items():
    #     print(f"{k:20}", '\t', 10**v.x)
    return


@app.cell
def _(df):
    set(df.index.get_level_values('set'))
    return


@app.cell
def _(
    df,
    model_gamma_log_mean,
    model_gamma_log_median,
    np,
    plt,
    res_power,
    res_rdn,
):
    dd_fig, _axs = plt.subplots(1,3, sharey=True)

    _X = df.loc[['edf', 'gdf']]['pre']
    _Y = df.loc[['edf', 'gdf']]['log_gamma']

    _x = np.linspace(_X.min(), _X.max(), 100)
    _pred_rdn = model_gamma_log_median(_x, res_rdn[f'log_gamma'].x)
    _pred_power = model_gamma_log_mean(_x, res_power[f'log_gamma'].x)

    _ax = _axs[0]
    _ax.scatter(_X, _Y)
    _ax.plot(_x, _pred_rdn)
    _ax.plot(_x, _pred_power)

    _ax.set_xscale('log')
    # _ax.set_title(f"{10**res_rdn[f'log_gamma_{dset}'].x[2]:.2f}")
    # _ax.plot(_dff[''])

    plt.show()
    return


@app.cell
def _(df, np, pd):
    def coverage(statistic, low_or_high):
        def f(x):
            x = x.sort_values()
            if statistic == 'median':
                med = x.median()
                if low_or_high == 'low': 
                    return med-x.iloc[14]
                else:
                    return x.iloc[25] - med
            elif statistic == 'iq':
                iq = x.quantile(0.75) - x.quantile(0.25)
                if low_or_high == 'low':
                    return iq - x.iloc[24] + x.iloc[15]
                else:
                    return x.iloc[34] - x.iloc[5] - iq

        return f


    def coverage_med(x):
        x = x.sort_values()

        return x.iloc[25] - x.iloc[14]

    def coverage_iq(x):
        x = x.sort_values()

        ci1 = x.iloc[15] - x.iloc[5]
        ci3 = x.iloc[34] - x.iloc[24]

        return np.sqrt(ci1**2 + ci3**2)


    cbins, _bins = pd.qcut(df['pre'], q=10, retbins=True)
    bins = np.convolve(_bins, 0.5*np.ones(2), mode='valid')
    return bins, cbins, coverage


@app.cell
def _(
    bins,
    cbins,
    coverage,
    deltak,
    deltan,
    df,
    e50,
    model_gamma_mean,
    model_gamma_median,
    model_gamma_q,
    np,
    plt,
    res_power,
    res_rdn,
    sbar,
):
    def _plotter(df, key):

        y = df[key].groupby(cbins).agg(
            q1 = lambda x: x.quantile(0.25),
            med = lambda x: x.quantile(0.5),
            q3 = lambda x: x.quantile(0.75),
            mean = 'mean',
            std = 'std',
        )

        ye = df[key].groupby(cbins).agg(
            med_l = coverage('median', 'low'),
            med_h = coverage('median', 'high'),
            iq_l = coverage('iq', 'low'),
            iq_h = coverage('iq', 'high'),
            mean = 'sem',
        )

        fig, axs = plt.subplots(3,3, figsize=(12,9))
        xx = np.linspace(0.12, 11, 100)

        ax = axs[0,0]
        ax.scatter(df['pre'], df[key], s=10, alpha=0.1, c='black', lw=0)
        ax.plot(bins, y['med'], color='black', lw=3, label='Data median')
        ax.plot(bins, y['q1'], color='black', lw=1, linestyle=(0,(8,3)), label='Data IQ')
        ax.plot(bins, y['q3'], color='black', lw=1, linestyle=(0,(8,3)))

        ax.plot(xx, model_gamma_median(xx, res_rdn['gamma'].x), color='tab:blue', lw=3, zorder=0, label='Model median')
        ax.fill_between(
            xx, 
            model_gamma_q(xx, res_rdn[key].x, 0.83),
            model_gamma_q(xx, res_rdn[key].x, -0.83),
            color='tab:blue', alpha=0.2,
            lw=0,
            label='Model IQ',
        )

        ax.legend(frameon=False, fontsize=9)
        ax.set_ylim(0.2, 5)


        ax = axs[0,1]
        ax.errorbar(bins, y['med'], yerr=ye[['med_l', 'med_h']].T, c='black', fmt='o', label='Data')
        ax.plot(xx, model_gamma_median(xx, res_rdn[key].x), color='tab:blue', lw=3, zorder=0, label='Model')
        ax.legend(frameon=False, fontsize=9)

        ax = axs[0,2]
        ax.errorbar(bins, y['q3'] - y['q1'], yerr=ye[['med_l', 'med_h']].T, c='black', fmt='o', label='Data')
        ax.plot(xx, model_gamma_q(xx, res_rdn[key].x, 0.83) - model_gamma_q(xx, res_rdn[key].x, -0.83), color='tab:blue', lw=3, zorder=0, label='Model')
        ax.legend(frameon=False, fontsize=9)

        ax = axs[1,1]
        ax.errorbar(bins, y['med'], yerr=ye[['med_l', 'med_h']].T, c='black', fmt='o', label='Data')
        ax.plot(
            xx, 
            model_gamma_median(xx, np.array((np.log10(deltak),np.log10(deltan),np.log10(-sbar), np.log10(e50*7021),))), 
            color='tab:blue', lw=3, zorder=0, label='Model'
        )

        ax.legend(frameon=False, fontsize=9)

        ax = axs[2,0]
        ax.scatter(df['pre'], df[key], s=10, alpha=0.1, c='black', lw=0)
        ax.plot(bins, y['mean'], color='black', lw=3, label='Data Mean')
        ax.plot(bins, y['mean']-y['std'], color='black', lw=1, linestyle=(0,(8,3)), label='Data SEM')
        ax.plot(bins, y['mean']+y['std'], color='black', lw=1, linestyle=(0,(8,3)))
        ax.plot(xx, model_gamma_mean(xx, res_power[key].x), color='tab:red', lw=3, zorder=0, label='Power model mean')
        ax.legend(frameon=False, fontsize=9)
        ax.set_ylim(0.2, 5)

        ax = axs[2,1]
        ax.errorbar(bins, y['mean'], yerr=ye['mean'].T, c='black', label='Data', fmt='o')
        ax.plot(xx, model_gamma_mean(xx, res_power[key].x), color='tab:red', lw=3, zorder=0, label='Power model')
        ax.legend(frameon=False, fontsize=9)

        axs[0,0].set_title('Plasticity response ratio')
        axs[0,1].set_title('Median of ratio')
        axs[0,2].set_title('IQ of the ratio')
        axs[2,0].set_title('Plasticity response ratio')
        axs[2,1].set_title('Mean of ratio')
        axs[2,2].remove()

        for ax in axs.flatten():
            ax.set_xlabel('Basal size')
            ax.set_ylabel('Post-basal ratio')
            ax.set_xlim(0.1,12)
            ax.set_xscale('log')
            # ax.set_yscale('log')

        fig.subplots_adjust(hspace=0.5, wspace=0.3)

        return axs


    _plotter(df, 'gamma')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
