import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    from functools import partial
    import marimo as mo
    import matplotlib.pyplot as plt
    import jax 
    import jax.numpy as jnp
    from jax.scipy.stats import norm, beta
    import numpy as np
    from statsmodels.stats.diagnostic import normal_ad
    from scipy.stats import normaltest, shapiro
    from tqdm import tqdm

    plt.style.use('default')
    return jax, jnp, mo, norm, normal_ad, normaltest, plt, shapiro


@app.cell
def _(jax, jnp, norm):
    def generate_z(key, alphain, alphaout):

        key_in, key_out = jax.random.split(key)
        kins = jax.random.beta(key_in, alphain, alphain, shape=(200,100,)) 
        kouts = jax.random.beta(key_out, alphaout, alphaout, shape=(200,100,)) 
        return jnp.log(kins/kouts)


    def test_nlpd(key, alphain, alphaout):
        X = generate_z(key, alphain, alphaout)
        mu_hats = X.mean(axis=1)
        var_hats = X.var(ddof=1, axis=1)

        return norm.logpdf(X, mu_hats[:,None], jnp.sqrt(var_hats[:,None])).sum(axis=1)

    # jax.config.update("jax_enable_x64", True)

    EDGES = jnp.linspace(-4.0, 4.0, 11)
    Q = jnp.diff(norm.cdf(EDGES)); Q = Q / Q.sum()

    def test_kl(key, alphain, alphaout):
        X = generate_z(key, alphain, alphaout)          # same samples as the tests
        Z = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, ddof=1, keepdims=True)
        counts = jax.vmap(lambda z: jnp.histogram(z, bins=EDGES)[0])(Z)
        p_hat = counts / counts.sum(axis=1, keepdims=True)
        return jax.scipy.special.kl_div(p_hat, Q[None, :]).sum(axis=1)

    return generate_z, test_kl


@app.cell
def _(jax, jnp, test_kl):
    N_cvs = 20

    key = jax.random.key(2026)
    subkeys = jax.random.split(key, num=(N_cvs,N_cvs,))

    cvs = jnp.linspace(0.01,0.9,N_cvs)
    alphas = jnp.array([0.5 * (1/cv**2 - 1) for cv in cvs])
    alphains, alphaouts = jnp.meshgrid(alphas, alphas)

    # vmap_f = jax.vmap(
    #     jax.vmap(
    #         test_nlpd,
    #         in_axes=(0,0,0)
    #     ),
    #     in_axes=(0,0,0,),
    # )
    # print('Computing nlpds...')
    # nlpds = vmap_f(subkeys, alphains, alphaouts,)
    # print('Done.')

    vmap_f = jax.vmap(
        jax.vmap(
            test_kl,
            in_axes=(0,0,0)
        ),
        in_axes=(0,0,0,),
    )
    print('Computing kls...')
    kls = vmap_f(subkeys, alphains, alphaouts,)
    print('Done.')
    return alphains, alphaouts, cvs, kls, subkeys


@app.cell
def _(
    alphains,
    alphaouts,
    generate_z,
    jnp,
    mo,
    normal_ad,
    normaltest,
    shapiro,
    subkeys,
):
    print('Computing p-vals...')
    ads = []
    shapiros = []
    gigidags = []

    for _key, ai, ao in mo.status.progress_bar(zip(subkeys.flatten(),
        alphains.flatten(), alphaouts.flatten()), total=400):
        X = generate_z(_key, ai, ao)
        ads.append(normal_ad(X, axis=1)[1])
        shapiros.append(shapiro(X, axis=1).pvalue)
        gigidags.append(normaltest(X, axis=1).pvalue)


    ads = jnp.array(ads).reshape(20,20,-1)
    shapiros = jnp.array(shapiros).reshape(20,20,-1)
    gigidags = jnp.array(gigidags).reshape(20,20,-1)
    return ads, gigidags, shapiros


@app.cell
def _(ads, cvs, gigidags, kls, plt, shapiros):
    fig, axs = plt.subplots(1,8,figsize=(18,3), width_ratios=(1,1,1,0.,0.05,0.1, 1, 0.05), dpi=300)
    axs[3].remove()
    axs[5].remove()

    contourf_kwargs = dict(levels=50, cmap='Blues')
    pmesh_kwargs = dict(cmap='Blues_r')
    contour_kwargs = dict(linestyle=(0,(20,5)))

    axs[0].pcolormesh(cvs, cvs, (ads>0.05).mean(axis=2), **pmesh_kwargs)
    axs[0].contour(cvs, cvs, (ads>0.05).mean(axis=2), levels=(0.5,), lw=3, colors='black')
    axs[0].contour(cvs, cvs, (ads>0.05).mean(axis=2), levels=(0.9,), lw=1, colors='black')
    axs[0].contour(cvs, cvs, (ads>0.05).mean(axis=2), levels=(0.1,), lw=1, colors='black')

    axs[1].pcolormesh(cvs, cvs, (shapiros>0.05).mean(axis=2), **pmesh_kwargs)
    axs[1].contour(cvs, cvs, (shapiros>0.05).mean(axis=2), levels=(0.5,), lw=3, colors='black')
    axs[1].contour(cvs, cvs, (shapiros>0.05).mean(axis=2), levels=(0.9,), lw=1, colors='black')
    axs[1].contour(cvs, cvs, (shapiros>0.05).mean(axis=2), levels=(0.1,), lw=1, colors='black')
    # _ = axs[2].contourf(cvs, cvs, (gigidags>0.05).mean(axis=2),**contourf_kwargs)

    _ = axs[2].pcolormesh(cvs, cvs, (gigidags>0.05).mean(axis=2), **pmesh_kwargs,)
    axs[2].contour(cvs, cvs, (gigidags>0.05).mean(axis=2), levels=(0.5,), lw=3, colors='black')
    axs[2].contour(cvs, cvs, (gigidags>0.05).mean(axis=2), levels=(0.9,), lw=1, colors='black')
    axs[2].contour(cvs, cvs, (gigidags>0.05).mean(axis=2), levels=(0.1,), lw=1, colors='black')
    plt.colorbar(_, cax=axs[4], ticks=(0.1, 0.5, 0.9))

    # _ = axs[6].pcolormesh(cvs, cvs, jnp.nan_to_num(nlpds, nan=-500).mean(axis=2), cmap='Greens_r')
    # axs[6].contour(cvs, cvs, nlpds.mean(axis=2), levels=(-300, -150, 0, 150), colors='white', linestyle=(0,(4,2)), lw=2)
    # plt.colorbar(_, cax=axs[-1], ticks=(-400,-200, 0, 200))

    _ = axs[6].pcolormesh(cvs, cvs, kls.mean(axis=2), cmap='Greens_r')
    axs[6].contour(cvs, cvs, kls.mean(axis=2), levels=(0.05, 0.1, 0.2), colors='black', linestyle=(0,(4,2)), lw=2)
    plt.colorbar(_, cax=axs[-1])
    # axs[6].set_xlim(0.01, 0.8)
    # axs[6].set_ylim(0.01, 0.8)

    for i, ax in enumerate(axs):
        if i==4 or i==7: continue
        ax.set_aspect('equal')
    
    plot_axs = axs[[0,1,2,6]]
    for ax in plot_axs:
        ax.set_xlabel(r'$CV_{in}$')
        ax.set_ylabel(r'$CV_{out}$')
    # fig.subplots_adjust(hspace=0.5)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
