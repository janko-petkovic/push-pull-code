import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import jax 
    import jax.numpy as jnp
    from jax.scipy.stats import norm, beta

    from statsmodels.stats.diagnostic import normal_ad

    plt.style.use('default')
    return jax, jnp, norm, normal_ad, plt


@app.cell
def _(jax, jnp, norm, normal_ad, plt):
    def generate_z(key, alphain, alphaout):

        key_in, key_out = jax.random.split(key)
        kins = jax.random.beta(key_in, alphain, alphain, shape=(10,10,)) 
        kouts = jax.random.beta(key_out, alphaout, alphaout, shape=(10,10,)) 
        return jnp.log(kins/kouts)

    def ad_wrapper(X):
        return jnp.array(normal_ad(X, axis=1), dtype=jnp.float32)[1]

    def test_ad(key, alphain, alphaout):
        X = generate_z(key, alphain, alphaout)
    
        return jax.pure_callback(
            ad_wrapper,
            jax.ShapeDtypeStruct((1000,), jnp.float32),
            X,
            vmap_method='expand_dims'
        )[1]

    def test_nlpd(key, alphain, alphaout):
        X = generate_z(key, alphain, alphaout)
        mu_hat = X.mean()
        var_hat = X.var(ddof=1)
    
        return jnp.log(norm.pdf(X, mu_hat, var_hat).sum())


    def _main():
        N_cvs = 20
        N_samples = 1000
    
        key = jax.random.key(2026)
        subkeys = jax.random.split(key, num=(N_cvs,N_cvs,))

        alphas = jnp.array([0.5 * (1/cv**2 - 1) for cv in jnp.linspace(0.01,0.4,N_cvs)])
        alphains, alphaouts = jnp.meshgrid(alphas, alphas)

        vmap_f = jax.vmap(
            jax.vmap(
                test_ad,
                in_axes=(0,0,0)
            ),
            in_axes=(0,0,0,),
        )

        nlpds = vmap_f(subkeys, alphains, alphaouts,)

        # print(nlpds.shape)
        plt.pcolormesh(nlpds)
        # plt.contourf(nlpds)
        # plt.colorbar()
        return nlpds


    res = _main()
    plt.show()

    # print(res)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
