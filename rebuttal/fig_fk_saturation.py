import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import least_squares

    return least_squares, np, plt


@app.cell
def _(np):
    X = np.array([0,2,10])
    Ns = np.array([1, 1/1.8, 1/2.5])
    Kb = np.array([1, 1.2, 1.8])
    Nb = np.array([1, 1/1.2, 1/1.8])

    X_plot = np.linspace(0,10,100)
    return Kb, Nb, Ns, X, X_plot


@app.cell
def _(Kb, Nb, Ns, X, X_plot, least_squares, np, plt):
    def model(x,p):
        return (1-p[0]) + p[0]*np.exp(-x/p[1])
    
    def residuals(p,x,y):
        return y - model(x,p)

    def eq_annotate(ax, loc, res):
        a = res.x[0]
        b = res.x[1]
    
        if a >= 0:
            ax.text(*loc, fr"$y = (1 - {a:.2f}) + {a:.2f} e^{{-x/{b:.2f}}}$" )
        else:
            ax.text(*loc, fr"$y = (1 + {np.abs(a):.2f}) {a:.2f} e^{{-x/{b:.2f}}}$" )

    res_ns = least_squares(residuals, np.array((1,1)), args=(X, Ns))
    res_nb = least_squares(residuals, np.array((1,1)), args=(X, Nb))
    res_kb = least_squares(residuals, np.array((1,1)), args=(X, Kb))


    fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(12,3), dpi=300)
    line_kwg = dict(lw=3,)
    sct_kwg = dict(s=70,)

    ax = axs[0]
    ax.scatter(X, Ns, **sct_kwg)
    ax.plot(X_plot, model(X_plot, res_ns.x), **line_kwg)
    eq_annotate(ax, (0.5, 1.7), res_ns)


    ax = axs[1]
    ax.scatter(X, Nb, **sct_kwg)
    ax.plot(X_plot, model(X_plot, res_nb.x), **line_kwg)
    eq_annotate(ax, (0.5, 1.7), res_nb)

    ax = axs[2]
    ax.scatter(X, Kb, **sct_kwg)
    ax.plot(X_plot, model(X_plot, res_kb.x), **line_kwg)
    eq_annotate(ax, (0.5, 0.5), res_kb)

    for ax in axs:
        ax.axhline(y=1, linestyle=(0,(10,4)), lw=1, c='black', zorder=-10)
        ax.set_xlabel(r'FK506 concentration $[\mu m]$')

    axs[0].set_ylabel('Relative amount')
    axs[0].set_title(r'Induced calcineurin $N_s$')
    axs[1].set_title(r'Basal calcineurin $N_b$')
    axs[2].set_title(r'Basal CaMKII $K_b$')
    plt.show()
    # plt.plot(X, Nb)
    # plt.plot(X, Kb)
    return


if __name__ == "__main__":
    app.run()
