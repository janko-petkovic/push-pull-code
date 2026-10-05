import jax.numpy as jnp
from typing import Iterable
import matplotlib.pyplot as plt
from numpy import expand_dims


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

    # ax = axs[2]
    # l,m,h = quick_mapper('ud')
    # ax.plot(dendrite_X, m, '-', color='gray')
    # ax.fill_between(dendrite_X, l, h, alpha=0.2, color='gray')
    # ax.set_ylim(y_lims['ud'])

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

def plot_comparison(
    fig,
    experiments,
    t_idxs: Iterable[int],
    x_view: Iterable[int] = [0,100],
    summary: str ="medianiq",
    relative: bool =True,
    y_lims = None,
    keys = Iterable[str]
    # **kwargs,
):
    '''n_experiments rows, t_idx cols'''

    def quick_mapper(key, experiment, t_idx):
        mask = dendrite_mask if 'd' in key else spine_mask
        return map(
            lambda x: x[t_idx, mask], 
            experiment.dataset.get_summary(key, relative=relative)
        )

    axs = fig.subplots(len(keys), len(t_idxs),sharex=True, sharey='row')
    colors = plt.cm.Blues(jnp.linspace(0,1,len(experiments)))
    colors=['black', 'tab:blue']


    if len(axs.shape) == 1: axs = axs[None,:]
    print(axs.shape)
    

    for row, key in zip(axs, keys):
        for color, experiment in zip(colors, experiments):
            _ = experiment.get_X_and_mask_from_view(x_view)
            dendrite_mask, dendrite_X, spine_mask, spine_X = _

            for ax, t_idx in zip(row, t_idxs):
                l,m,h = quick_mapper(key, experiment, t_idx)
                ax.plot(spine_X, m, '-o', color=color, lw=2)
                ax.plot(spine_X, l, '-', color=color,lw=0.5)
                ax.plot(spine_X, h, '-', color=color,lw=0.5)
                ax.fill_between(spine_X, l, h, alpha=0.2, color=color)
                # ax.set_ylim(y_lims[key])

                for x in experiment.xs_stim:
                    ax.axvline(x, color='tab:orange', lw=1, linestyle='--',
                               zorder=-10)

