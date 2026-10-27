from functools import partial
import jax.numpy as jnp
from typing import Sequence
import matplotlib.pyplot as plt
from matplotlib import ticker

from rdn.rebuttal.experiment import Result



def consistency_plot(experiment, ax):
    print(experiment.dataset.yt["ks"].shape)
    print(experiment.dataset.yt["ud"].shape)
    pis = jnp.stack(
        [experiment.dataset.yt[k].sum(axis=-1) for k in ["ud", "us", "ps"]]
    ).sum(axis=0)

    for pi in pis:
        ax.plot(pi / pi[0], c="black", lw=1)


def plot_snapshot(
    fig,
    experiment,
    t_idx,
    x_view: list = [0,100],
    summary="medianiq",
    relative=True,
    y_lims = dict(ps=(0.9, 1.5),ksns=(0, 20),ud=(0.9, 1.1)),
    # **kwargs,
) -> None:

    def quick_mapper(key):
        mask = dendrite_mask if 'd' in key else spine_mask
        return map(
            lambda x: x[t_idx, mask], 
            experiment.dataset.get_summary(key, relative=relative)
        )


    _ = experiment.get_X_and_mask_from_view(x_view)
    dendrite_mask, dendrite_X, spine_mask, spine_X = _

    axs = fig.subplots(3,1,sharex=True)

    # breakpoint()
    ax = axs[0]
    l,m,h = quick_mapper('ps')
    ax.plot(spine_X, m, '-o')
    ax.fill_between(spine_X, l, h, alpha=0.2)
    ax.set_ylim(y_lims['ps'])

    ax = axs[1]
    l,m,h = quick_mapper('ks')
    ax.bar(spine_X-0.15, m, width=0.3, yerr=(m-l, h-m), ecolor='forestgreen', color='forestgreen')
    l,m,h = quick_mapper('ns')
    ax.bar(spine_X+0.15, m, width=0.3, yerr=(m-l, h-m), ecolor='mediumvioletred', color='mediumvioletred')
    ax.set_ylim(y_lims['ksns'])

    for ax in axs:
        for x in experiment.xs_stim:
            ax.axvline(x)


def plot_series(
    experiment,
    ax,
    key,
    x_idxs,
    t_slice=slice(0, None, None),
    summary="medianiq",
    relative=True,
    **kwargs,
) -> None:

    if not hasattr(x_idxs, "__iter__"):
        x_idxs = (x_idxs,)
    x_idxs = jnp.array(x_idxs)

    full_lmh = experiment.dataset.get_summary(key, summary, relative)
    lmh = []
    for x in full_lmh:
        lmh.append(x[t_slice, x_idxs].T)

    X = experiment.ts[t_slice]
    marker = ""

    for l, m, h in zip(*lmh):
        ax.plot(X, m, c=kwargs.get("color"), lw=1, marker=marker)
        ax.fill_between(X, l, h, alpha=0.1, color=kwargs.get("color"))


def plot_field(
    ax, experiment, key, x_view, summary="medianiq", relative=False, **kwargs
) -> None:

    if "d" in key:
        mask = (
            (experiment.xs_dendrite >= x_view[0])*(experiment.xs_dendrite < x_view[1])
        )
        X = experiment.xs_dendrite[mask]

    else:
        mask = (experiment.xs_spine >= x_view[0])*(experiment.xs_spine < x_view[1])
        X = experiment.xs_spine[mask]

    l, m, h = experiment.dataset.get_summary(key, summary, relative)

    if "d" in key:
        X = experiment.xs_dendrite
    else:
        X = experiment.xs_spine

    sums = experiment.dataset.get_summary(key, summary, relative)
    _, med, _ = map(lambda x: x[:, mask], sums)

    ax.imshow(med, cmap="berlin_r", vmin=0.5, vmax=1.1)



def plot_cool(
    ax,
    experiment,
    key,
    t_idx,
    x_view: list = [0,100],
    summary="medianiq",
    relative=True,
    showstim=False,
    **kwargs,
) -> None:
    '''this I used for Tatjana's visualization, not super informative'''

    if "d" in key:
        mask = (
            (experiment.xs_dendrite >= x_view[0])*(experiment.xs_dendrite < x_view[1])
        )
        X = experiment.xs_dendrite[mask]
        marker='-'
        lw=2
    else:
        mask = (experiment.xs_spine >= x_view[0])*(experiment.xs_spine < x_view[1])
        X = experiment.xs_spine[mask]
        marker='-o'
        lw=0.5

    Y = experiment.dataset.yt[key][:, t_idx, mask]

    if relative:
        Y /= experiment.dataset.y0[key][:, 0, mask]
    
    smr = experiment.dataset.get_summary(key, summary, relative)
    _, med, _ = map(lambda x: x[t_idx, mask], smr)

    ax.plot(X, med, marker, color='white', lw=lw, markersize=7,
        label=kwargs.get('label'))
    ax.axhline(1,  color='white', lw=0.5, linestyle=(0,(20,10)))

    for y in Y:
        ax.plot(X, y, color='white', lw=0.1, markersize=5, alpha=1,
                mew=0)

    if showstim:
        ax.vlines(experiment.xs_stim, 3, 3.5, lw=2)

    ax.legend(frameon=False, loc='upper left')
    ax.set_ylabel('Post-pre ratio')
    ax.set_xlabel(r'Position along dendrite $[\mu m]$')


def _compact_plot_comparison(
    fig,
    experiments: Sequence[Result],
    t_idxs: Sequence[int],
    x_view: Sequence[int] = [0,100],
    summary: str = "medianiq",
    relative: bool = True,
    y_lims: dict = dict(ps=(0.5, 2.5), ksns=(0, 20), ud=(0.9, 1.1)),
    key: str = 'ps',
):
    '''n_experiments rows, t_idx cols'''

    def quick_mapper(key, experiment, t_idx):
        mask = dendrite_mask if 'd' in key else spine_mask
        return map(
            lambda x: x[t_idx, mask], 
            experiment.dataset.get_summary(key, relative=relative)
        )

    axs = fig.subplots(1, len(t_idxs), sharex=True, sharey=True,)

    colors = plt.cm.Blues(jnp.linspace(0,1,len(experiments)))
    colors=['tab:blue', 'black']
    
    # This logic is not the best, but we implement it this way for this
    # time
    for ax, t_idx in zip(axs, t_idxs):
        _ = experiments[0].get_X_and_mask_from_view(x_view)
        dendrite_mask, dendrite_X, spine_mask, spine_X = _

        ld,md,hd = quick_mapper(key, experiments[0], t_idx)
        lq,mq,hq = quick_mapper(key, experiments[1], t_idx)

        ax.scatter(spine_X, md, color=colors[0], s=10,)
        ax.plot(spine_X, ld, color=colors[0],lw=0.5)
        ax.plot(spine_X, hd, color=colors[0],lw=0.5)
        ax.fill_between(spine_X, ld, hd, alpha=0.2, color=colors[0], lw=0)

        ax.scatter(spine_X, mq, s=10, color=colors[1])
        ax.plot(spine_X, lq, '-', color=colors[1],lw=1)
        ax.plot(spine_X, hq, '-', color=colors[1],lw=1)
        ax.fill_between(spine_X, lq, hq, facecolor="none", hatch='//////',
                        hatchcolor=colors[1], hatch_linewidth=0.5,
                        ec="none")

        ax.set_title(f'{t_idx} min')

        ax.set_xticks((42,50,58))
        ax.set_xlabel("Spine position [um]")

        for x in experiments[0].xs_stim:
                ax.axvline(x, color='tab:orange', lw=1, zorder=-10)

    axs[0].set_ylabel('Response\nratio', weight='bold', labelpad=10)

    for ax in axs[1:].flatten():
        ax.set_ylim(-25,25)

    fig.subplots_adjust(wspace=0.1)


def plot_comparison(
    fig,
    experiments: Sequence[Result],
    t_idxs: Sequence[int],
    x_view: Sequence[int] = [0,100],
    summary: str ="medianiq",
    relative: bool =True,
    key: str = 'ps',
    compact: bool = False,
    **kwargs,
):
    '''n_experiments rows, t_idx cols'''

    if compact:
        _compact_plot_comparison(
            fig,
            experiments,
            t_idxs,
            x_view,
            summary,
            relative,
            key,
        )
        return None 

    def quick_mapper(key, experiment, t_idx):
        mask = dendrite_mask if 'd' in key else spine_mask
        return map(
            lambda x: x[t_idx, mask], 
            experiment.dataset.get_summary(key, relative=relative)
        )

    axs = fig.subplots(4, len(t_idxs),
                       sharex=True, sharey='row',
                       height_ratios=(2,1,1,1))
    if kwargs.get('colors') is None:
        colors=['black', 'tab:blue']
    else:
        colors = kwargs.get('colors')
    
    # This logic is not the best, but we implement it this way for this
    # time
    for col, t_idx in zip(axs.T, t_idxs):
        _ = experiments[0].get_X_and_mask_from_view(x_view)
        dendrite_mask, dendrite_X, spine_mask, spine_X = _

        ld,md,hd = quick_mapper(key, experiments[0], t_idx)
        lq,mq,hq = quick_mapper(key, experiments[1], t_idx)

        iqld, iqhd = md - ld, hd - md
        iqlq, iqhq = mq - lq, hq - mq

        ax = col[0]
        ax.scatter(spine_X, md, color=colors[0], s=20,)
        # ax.plot(spine_X, ld, color=colors[0],lw=0.5, linestyle=(0,(8,4)))
        # ax.plot(spine_X, hd, color=colors[0],lw=0.5, linestyle=(0,(8,4)))
        # ax.plot(spine_X, ld, color=colors[0],lw=0.5)
        # ax.plot(spine_X, hd, color=colors[0],lw=0.5)
        # ax.fill_between(spine_X, lq, hq, facecolor="none", hatch=r'\\\\\\',
        #                 hatchcolor=colors[0], hatch_linewidth=0.5,
        #                 ec="none")
        ax.fill_between(spine_X, ld, hd, alpha=0.4, color=colors[0], lw=0)

        ax.scatter(spine_X, mq, s=20, color=colors[1])
        ax.plot(spine_X, lq, '-', color=colors[1],lw=2)
        ax.plot(spine_X, hq, '-', color=colors[1],lw=2)
        ax.fill_between(spine_X, lq, hq, facecolor="none", hatch='//////',
                        hatchcolor=colors[1], hatch_linewidth=0.5,
                        ec="none")

        ax = col[1]
        ax.plot(spine_X, (mq-md)/md*100, '-', color='black', lw=3,
                markersize=5)

        ax = col[2]
        ax.plot(spine_X, (iqhq-iqhd)/md*100, '-', color='black', lw=2,
                markersize=5)

        ax = col[3]
        ax.plot(spine_X, (iqlq-iqld)/md*100, '-', color='black', lw=2,
                markersize=5)

        col[0].set_title(f'{t_idx} min')
        col[0].set_yticks((1,1.5))

        col[-1].set_xticks((42,50,58))
        col[-1].set_xlabel(r"Spine position [$\mu m$]")

    for x in experiments[0].xs_stim:
        for ax in axs.flatten():
            ax.axvline(x, color='tab:orange', lw=1,
                   zorder=-10)

    for ax in axs[1:].flatten():
        ax.axhline(y=0, linestyle='--', lw=1, c='black')
        ax.grid(visible=True, axis='y')
        ax.set_ymargin(0.5)
        ax.set_ylim(kwargs.get('y_lims'))
        if kwargs.get('y_ticks') is not None:
            ax.set_yticks(kwargs.get('y_ticks'))

    col = axs.T[0]

    col[0].set_ylabel('Response\nratio', weight='bold', labelpad=10)
    col[1].set_ylabel('Med')
    col[2].set_ylabel('IQ 2')
    col[3].set_ylabel('IQ 3')
    col[3].text(32, 0, 'Relative\ndifference [%]', weight='bold',
                rotation=90, ha='center'
                )

    fig.subplots_adjust(wspace=0.1)


def plot_bartol(
    fig,
    experiment: Sequence[Result],
    **kwargs,
):
    '''n_experiments rows, t_idx cols'''

    import pandas as pd
    import numpy as np
    import seaborn as sns
    from scipy.stats import linregress

    timepoints =[t.item() for t in experiment.ts]
    sessions = [s.item() for s in jnp.arange(experiment.dataset.yt['ps'].shape[0])]

    multi_index = pd.MultiIndex.from_product(
        (sessions, timepoints), names=['session', 't'],
    )

    stim_indexes = np.array(experiment.xs_stim)
    test_indexes = np.array([0,-1])
    QUERY_STR = 't == 299'

    stim_srs = np.stack([experiment.dataset.yt['ps'][:,:, si] for si in
                         stim_indexes])
    test_srs = np.stack([experiment.dataset.yt['ps'][:,:, si] for si in
                         test_indexes])

    srs = np.concat((stim_srs, test_srs), axis=0)

    df = pd.DataFrame(
        np.array([s[*np.stack(multi_index.values).T] for s in srs]).T,
        columns=['a','b','ta','tb'],
        index=multi_index
    )

    df = df.query(QUERY_STR)

    for col in df.columns:
        df[f'log_{col}'] = np.log(df[col])

    # ORDERING
    m = df['a'] >= df['b']
    df['first'] = np.where(m, df['a'], df['b']) 
    df['second'] = np.where(m, df['b'], df['a']) 
    df['log_first'] = np.where(m, df['log_a'], df['log_b']) 
    df['log_second'] = np.where(m, df['log_b'], df['log_a']) 
    plot_min, plot_max = df['log_second'].min(), df['log_first'].max()


    m = df['ta'] >= df['tb']
    df['test_first'] = np.where(m, df['ta'], df['tb']) 
    df['test_second'] = np.where(m, df['tb'], df['ta']) 
    df['log_test_first'] = np.where(m, df['log_ta'], df['log_tb']) 
    df['log_test_second'] = np.where(m, df['log_tb'], df['log_ta']) 
    plot_test_min, plot_test_max = (
        df['log_test_second'].min(), df['log_test_first'].max()
    )

    sirpe = linregress(
        df['log_first'], 
        df['log_second']
    )

    test_sirpe = linregress(
        df['log_test_first'], 
        df['log_test_second'], 
    )

    print(sirpe.rvalue**2)
    print(test_sirpe.rvalue**2)
    print(sirpe.pvalue)
    print(test_sirpe.pvalue)

    axs = fig.subplots(1,2)

    ax = sns.regplot(
        df,
        x='log_first',
        y='log_second',
        ax=axs[0],
        color='tab:blue',
    )

    ax.plot(
        (plot_min, plot_max),
        (plot_min, plot_max),
        c='black',
        linestyle=(0,(10,4)),
        lw=1,
    )

    ax.text(
        0.14,0.65, 
        (
            fr'$y = {sirpe.slope:.2f}*x - {np.abs(sirpe.intercept):.2f}$'
            +'\n'
            +fr'$r^2 = {sirpe.rvalue**2:.2f}$'
            +'\n'
            +fr'$n = {max(sessions)+1}$'
        ),
        transform=fig.transFigure,
    )
    ax.set_title('Stimulated spines')
    
    ax = sns.regplot(
        df,
        x='log_test_first',
        y='log_test_second',
        color='black',
        ax=axs[1]
    )

    ax.plot(
        (plot_test_min, plot_test_max),
        (plot_test_min, plot_test_max),
        c='black',
        linestyle=(0,(10,4)),
        lw=1,
    )

    ax.text(
        0.6,0.65, 
        (
            fr'$y = {test_sirpe.slope:.2f}*x + {test_sirpe.intercept:.2f}$'
            +'\n'
            +fr'$r^2 = {test_sirpe.rvalue**2:.2f}$'
            +'\n'
            +fr'$n = {max(sessions)+1}$'
        ),
        transform=fig.transFigure,
    )

    ax.set_title('Non-stimulated spines')

    for ax in axs:
        ax.set_xmargin(0.1)
        ax.set_ymargin(0.1)
        ax.set_xlabel('log spine size 1')
        ax.set_ylabel('log spine size 2')

    fig.subplots_adjust(wspace=0.5, bottom=0.2)
