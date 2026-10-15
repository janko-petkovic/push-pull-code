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
    from tqdm import tqdm

    plt.style.use('default')
    return jax, jnp, mo, norm, normal_ad, plt


@app.cell
def _(jax, jnp, norm):
    def generate_z(key, alphain, alphaout):

        key_in, key_out = jax.random.split(key)
        kins = jax.random.beta(key_in, alphain, alphain, shape=(100,100,)) 
        kouts = jax.random.beta(key_out, alphaout, alphaout, shape=(100,100,)) 
        return jnp.log(kins/kouts)


    def test_nlpd(key, alphain, alphaout):
        X = generate_z(key, alphain, alphaout)
        mu_hats = X.mean(axis=1)
        var_hats = X.var(ddof=1, axis=1)

        return norm.logpdf(X, mu_hats[:,None], jnp.sqrt(var_hats[:,None])).sum(axis=1)


    def test_kl(key, alphain, alphaout):
        z_key, gauss_key = jax.random.split(key)
        X = generate_z(z_key, alphain, alphaout)
        mu_hats = X.mean(axis=1)
        std_hats = jnp.sqrt(X.var(ddof=1, axis=1))

        gauss_samples = jax.random.normal(gauss_key, shape=X.shape)*std_hats[:, None] + mu_hats[:, None]

        return jax.vmap(
            jax.scipy.special.kl_div,
            in_axes=(0,0,)
        )(X, gauss_samples)




    return generate_z, test_kl


@app.cell
def _(generate_z, jax, jnp, mo, normal_ad, test_kl):
    N_cvs = 20

    key = jax.random.key(2026)
    subkeys = jax.random.split(key, num=(N_cvs,N_cvs,))

    cvs = jnp.linspace(0.01,0.4,N_cvs)
    alphas = jnp.array([0.5 * (1/cv**2 - 1) for cv in cvs])
    alphains, alphaouts = jnp.meshgrid(alphas, alphas)

    vmap_f = jax.vmap(
        jax.vmap(
            test_kl,
            in_axes=(0,0,0)
        ),
        in_axes=(0,0,0,),
    )

    print('Computing nlpds...')
    nlpds = vmap_f(subkeys, alphains, alphaouts,)
    print('Done.')

    print('Computing p-vals...')

    ads = []
    for key, ai, ao in mo.status.progress_bar(zip(subkeys.flatten(),
        alphains.flatten(), alphaouts.flatten()), total=400):
        X = generate_z(key, ai, ao)
        ads.append(normal_ad(X, axis=1)[0])

    ads = jnp.array(ads).reshape(20,20,-1)
    # plt.show()

    # print(res)
    return ads, cvs, nlpds


@app.cell
def _(jax):
    a = jax.random.normal(jax.random.key(1), shape=100)
    b = a+1
    jax.scipy.special.rel_entr(a,b)
    return


@app.cell
def _(ads, cvs, nlpds, plt):
    fig, axs = plt.subplots(1,3,figsize=(12,3))



    axs[0].contourf(cvs, cvs, ads.mean(axis=2))
    axs[1].contourf(cvs, cvs, nlpds.mean(axis=2))
    axs[2].contourf(cvs, cvs, ads.mean(axis=2))
    axs[2].contour(cvs, cvs, nlpds.mean(axis=2))

    # for ax in axs:
    #     ax.set_xticklabels(cvs)
    #     ax.set_yticklabels(cvs)

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
