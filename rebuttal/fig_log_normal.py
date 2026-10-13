import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import jax 
    import jax.numpy as jnp
    from jax.scipy.stats import norm

    plt.style.use('default')
    return jax, jnp, norm, plt


@app.cell
def _(norm):
    dir(norm)
    return


@app.cell
def _(jax, plt):
    kins, kouts = jax.random.beta(jax.random.key(20), 2,2, shape=(2, 10000)) * 1

    plt.hist(kins, density=True, bins=30)
    return


@app.cell
def _(jax, jnp, norm, plt):
    def _f(key, ss, offset):
        kins, kouts = (
            jax.random.beta(key, 15,15, shape=(2, 10000)) -offset[:,None]
        ) * ss[:, None] +  offset[:, None]

        print(kins.std()/kins.mean())

        X = jnp.log(kins/kouts)
        return X


    def _main():
        key = jax.random.key(2026)
        ss = jnp.array((0.5,0.2))
        offset = jnp.array((0.5,0.5))
        X = _f(key, ss=ss, offset=offset)
    
        mu_hat = X.mean()
        var_hat = X.var(ddof=1)
    
        plt.hist(X, bins=30, density=True)
        plt.scatter(X, norm.pdf(X, mu_hat, var_hat))

        nltd = jnp.log(norm.pdf(X, mu_hat, var_hat)).sum()

    _main()
    plt.show()


    return


if __name__ == "__main__":
    app.run()
