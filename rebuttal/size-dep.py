import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams["font.family"] = "open sans"

    from scipy.optimize import least_squares

    from rdn.validation import Simulation
    from rdn.defaults import pardict_from_result
    from rdn.fitting.models import LocalGaussModelTilde

    return (
        LocalGaussModelTilde,
        least_squares,
        np,
        pardict_from_result,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Set up the model parameters
    """)
    return


@app.cell
def _(LocalGaussModelTilde, pardict_from_result):
    model = LocalGaussModelTilde()

    model_p_dict = pardict_from_result(
        "output/multi_fitting/Multi_LocalGaussModelTilde/"
        "NLLAdast/1_3_5_7_Spine_data_fides_1200.hdf5",
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
    return (model_p_dict,)


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
def _(model_p_dict, np):
    mpd = model_p_dict

    muk = mpd['mu_log_K_N'][0]
    sigmak = np.sqrt(mpd['cov_log_K_N'][0,0])
    mun = mpd['mu_log_K_N'][1]
    sigman = np.sqrt(mpd['cov_log_K_N'][1,1])
    deltak = mpd['Ks']
    deltan = mpd['Ns']
    pi = mpd['Pi']


    rho = mpd['cov_log_K_N'][0,1] / sigmak / sigman
    sbar = (rho * sigmak - sigman) / (sigmak**2 + sigman**2 - 2 * rho * sigmak * sigman)

    e50 = np.exp(-mun * (1 - sbar) - muk * sbar)
    return deltak, deltan, e50, muk, mun, pi, rho, sbar, sigmak, sigman


@app.cell
def _(df, least_squares, np):
    def model_gamma_median(x,p):
        p = 10**p
        # p = pp
        deltak = p[0]
        deltan = p[1]
        sbar = p[2]
        e50 = p[3]

        return (
            (1 + deltak * e50 * x**(sbar - 1))
            / (1 + deltan * e50 * x**(sbar))
        )

    def model_gamma_log_median(x,p):
        return np.log(model_gamma_median(x,p))

    def model_gamma_q(x,p,f):
        p = 10**p
        # p = pp
        deltak = p[0]
        deltan = p[1]
        sbar = p[2]
        e50 = p[3] * np.exp(f)

        return (
            (1 + deltak * e50 * x**(sbar - 1))
            / (1 + deltan * e50 * x**(sbar))
        )

    def model_gamma_mean(x, p):
        p = 10**p
        return 1 + p[0] * x ** (-p[1])

    def model_gamma_log_mean(x, p):
        return np.log(model_gamma_mean(x,p))


    def residual(p, X, Y, model):
        return Y - model(X, p)

    res_rdn = {
        'log_gamma' : least_squares(residual, -np.ones(4), args=(df.loc[['edf','gdf']]['stz_pre'], df.loc[['edf','gdf']]['log_gamma'], model_gamma_log_median), loss='linear'),
        'log_gamma_gdf' : least_squares(residual, -np.ones(4), args=(df.loc[['gdf']]['stz_pre'], df.loc[['gdf']]['log_gamma'], model_gamma_log_median), loss='linear'),
        'log_gamma_bdf' : least_squares(residual, -np.ones(4), args=(df.loc[['bdf']]['stz_pre'], df.loc[['bdf']]['log_gamma'], model_gamma_log_median), loss='linear'),
        'log_gamma_edf' : least_squares(residual, -np.ones(4), args=(df.loc[['edf']]['stz_pre'], df.loc[['edf']]['log_gamma'], model_gamma_log_median), loss='linear'),
    }

    res_power = {
        'log_gamma' : least_squares(residual, -np.ones(4), args=(df.loc[['edf','gdf']]['stz_pre'], df.loc[['edf','gdf']]['log_gamma'], model_gamma_log_mean), loss='linear'),
        'log_gamma_g' : least_squares(residual, -np.ones(4), args=(df.loc[['gdf']]['stz_pre'], df.loc[['gdf']]['log_gamma'], model_gamma_log_mean), loss='linear'),
        'log_gamma_b' : least_squares(residual, -np.ones(4), args=(df.loc[['bdf']]['stz_pre'], df.loc[['bdf']]['log_gamma'], model_gamma_log_mean), loss='linear'),
        'log_gamma_e' : least_squares(residual, -np.ones(4), args=(df.loc[['edf']]['stz_pre'], df.loc[['edf']]['log_gamma'], model_gamma_log_mean), loss='linear'),
    }

    print('RDN')
    print('---------')
    for k, v in res_rdn.items():
        print(f"{k:20}", '\t', 10**v.x)

    print('')
    print('Power law')
    print('---------')
    for k, v in res_power.items():
        print(f"{k:20}", '\t', 10**v.x)
    return (
        model_gamma_log_mean,
        model_gamma_log_median,
        model_gamma_mean,
        model_gamma_median,
        model_gamma_q,
        res_power,
        res_rdn,
    )


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
    _fig, _axs = plt.subplots(1,3, sharey=True)

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
def _(np, plt):
    xx = np.linspace(-1,1,100)
    plt.scatter(xx+5,xx)
    plt.xscale('log')
    plt.show()
    return


@app.cell
def _(res_rdn):
    res_rdn['gamma']
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


@app.cell
def _(muk, mun, np, pi, rho, sbar, sigmak, sigman):
    (np.exp(muk - mun + (sigmak**2 + sigman**2 - 2 * rho * sigmak * sigman)/2) * 100 / pi)**sbar
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plotting parameters
    """)
    return


@app.cell
def _():
    # cbins, _bins = pd.qcut(X, q=10, retbins=True)
    # _bins = (_bins[1:] + _bins[:-1]) / 2

    # def coverage_median(x):
    #     return (
    #         x[x.rank(method="first") == 25].item()
    #         - x[x.rank(method="first") == 14].item()
    #     ) / 2

    # def coverage_iq(x):
    #     x = 
    #     ci1 = 
    #     eq1 = (
    #         x[x.rank(method="first") == 15].item()
    #         - x[x.rank(method="first") == 5].item()
    #     ) / 2

    #     eq3 = (
    #         x[x.rank(method="first") == 34].item()
    #         - x[x.rank(method="first") == 24].item()
    #     ) / 2

    #     eiq = np.sqrt(eq1**2 + eq3**2)
    #     return eiq

    # y = Y.groupby(cbins).agg(
    #     median="median",
    #     q1=lambda x: x.quantile(0.25),
    #     q3=lambda x: x.quantile(0.75),
    #     mean="mean",
    #     n=lambda x: len(x),
    # )
    # coverage = Y.groupby(cbins).agg(
    #     median=lambda x: coverage_median(x),
    #     iq=lambda x: coverage_iq(x),
    #     mean="sem",
    # )
    # med, q1, q3, mean, n = [y.iloc[:, i] for i in range(len(y.T))]
    # xx = np.linspace(0.03, 1, 100)
    # p0 = np.ones(4)
    # res_rdn = least_squares(
    #     residual, x0=p0, args=(X, Y, rdn_model), loss="soft_l1"
    # )
    # res_power = least_squares(
    #     residual, x0=p0, args=(X, Y, power_model), loss="linear"
    # )
    # med_res = np.array(res_rdn.x).copy()
    # lq_res = med_res.copy()
    # hq_res = med_res.copy()
    # lq_res[0] = lq_res[0] / 1.6
    # lq_res[2] = lq_res[2] / 1.6
    # hq_res[0] = hq_res[0] * 1.6
    # hq_res[2] = hq_res[2] * 1.6
    # _fig, _axs = plt.subplots(2, 3, figsize=(12, 6), dpi=200)
    # _ax = _axs[0, 0]
    # _ax.scatter(X, Y, s=1, alpha=0.1, c="black")
    # _ax.plot(_bins, med, c="black", zorder=10, lw=3, label="Data median")
    # _ax.plot(_bins, q1, c="black", linestyle=(0, (8, 2)), label="Data IQ")
    # _ax.plot(_bins, q3, c="black", linestyle=(0, (8, 2)))
    # _ax.plot(
    #     xx,
    #     rdn_model(xx, res_rdn.x),
    #     lw=4,
    #     c="tab:blue",
    #     zorder=0,
    #     label="Median model",
    # )
    # _ax.fill_between(
    #     xx,
    #     rdn_model(xx, lq_res),
    #     rdn_model(xx, hq_res),
    #     alpha=0.2,
    #     label="IQ model",
    # )
    # _ax.legend(frameon=False, fontsize=9)
    # _ax.set_ylim(0.5, 20)
    # _ax.set_title("Plasticity response ratio")
    # _ax = _axs[0, 1]
    # _ax.errorbar(
    #     _bins,
    #     med,
    #     yerr=coverage["median"].to_numpy(),
    #     fmt="o",
    #     c="black",
    #     label="Data median",
    # )
    # _ax.plot(xx, rdn_model(xx, med_res), lw=4, zorder=0, label="Median model")
    # _ax.set_title("Median of the ratio")
    # _ax.legend(frameon=False, fontsize=9)
    # _ax = _axs[0, 2]
    # _ax.errorbar(
    #     _bins,
    #     q3 - q1,
    #     yerr=coverage["iq"].to_numpy(),
    #     fmt="o",
    #     c="black",
    #     label="Data IQ",
    # )
    # _ax.plot(
    #     xx,
    #     rdn_model(xx, lq_res) - rdn_model(xx, hq_res),
    #     lw=4,
    #     zorder=0,
    #     label="IQ model",
    # )
    # _ax.legend(frameon=False, fontsize=9)
    # _ax.set_title("IQ range of the ratio")
    # _ax = _axs[1, 0]
    # _ax.scatter(X, Y, s=1, alpha=0.1, c="black")
    # _ax.plot(_bins, mean, c="black", zorder=10, lw=3, label="Data mean")
    # _ax.plot(
    #     _bins,
    #     mean - coverage["mean"] * 3,
    #     c="black",
    #     linestyle=(0, (8, 2)),
    #     label="Data SEM",
    # )
    # _ax.plot(
    #     _bins, mean + coverage["mean"] * 3, c="black", linestyle=(0, (8, 2))
    # )
    # _ax.plot(
    #     xx,
    #     power_model(xx, res_power.x),
    #     lw=4,
    #     c="gray",
    #     zorder=1,
    #     label="Mean model",
    # )
    # _ax.set_ylim(0.5, 20)
    # _ax.legend(frameon=False, fontsize=9)
    # _ax = _axs[1, 1]
    # _ax.errorbar(
    #     _bins,
    #     mean,
    #     yerr=coverage["mean"].to_numpy(),
    #     fmt="o",
    #     c="black",
    #     label="Data mean",
    # )
    # _ax.plot(
    #     xx,
    #     power_model(xx, res_power.x),
    #     lw=4,
    #     c="gray",
    #     zorder=1,
    #     label="Mean model",
    # )
    # _ax.legend(frameon=False, fontsize=9)
    # for _ax in _axs.flatten():
    #     _ax.set_yscale("log")
    #     _ax.set_xlabel("Normalized basal size")
    #     _ax.set_ylabel("Post-basal ratio")
    # _fig.subplots_adjust(hspace=0.5, wspace=0.5)

    # plt.show()
    return


if __name__ == "__main__":
    app.run()
