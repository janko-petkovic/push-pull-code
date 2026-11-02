import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    from dataclasses import replace
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    import rdn.rebuttal.diffusion_uptake as du
    from scipy.optimize import least_squares
    from scipy.integrate import trapezoid

    return du, least_squares, np, plt, replace, tqdm, trapezoid


@app.cell
def _(np):
    def model_gauss(x,p):
        return p[0] * np.exp(-x**2/p[1]**2)

    def model_expon(x,p):
        return p[0] * np.exp(-np.abs(x)/p[1])

    def residual(p, x, y, model):
        return y - model(x,p)

    return model_expon, model_gauss, residual


@app.cell
def _(du):
    orig_params = du.Params()
    return (orig_params,)


@app.cell
def _(
    du,
    least_squares,
    model_expon,
    model_gauss,
    np,
    orig_params,
    replace,
    residual,
    tqdm,
):
    res_gauss = dict(K={}, N={})
    res_expon = dict(K={}, N={})

    solutions = {}

    influxes = np.linspace(50, 1000, 5)

    for influx in tqdm(influxes):
        params = replace(orig_params, influx = influx)
        model = du.Model(params)

        _solution = model.simulate()

        solutions[influx] = _solution

        res_gauss['K'][influx] = least_squares(
            residual,
            x0=(1.,1.,),
            args=(_solution.x, _solution.Km[:,-1], model_gauss ),
        )

        res_gauss['N'][influx] = least_squares(
            residual,
            x0=(1.,1.,),
            args=(_solution.x, _solution.Nm[:,-1], model_gauss ),
        )

        res_expon['K'][influx] = least_squares(
            residual,
            x0=(1.,1.,),
            args=(_solution.x, _solution.Km[:,-1], model_expon ),
        )

        res_expon['N'][influx] = least_squares(
            residual,
            x0=(1.,1.,),
            args=(_solution.x, _solution.Nm[:,-1], model_expon ),
        )

    return influxes, res_expon, res_gauss, solutions


@app.cell
def _(
    influxes,
    model_expon,
    model_gauss,
    np,
    res_expon,
    res_gauss,
    solutions,
    tqdm,
):
    def rrmse(x,y):
        return np.sqrt(np.mean((x - y)**2))/y.std()

    rmse_gauss = dict(K={}, N={})
    rmse_expon = dict(K={}, N={})

    for _rgk, _rek, _rgn, _ren, _solution, _influx in tqdm(zip(
        res_gauss['K'].values(),
        res_expon['K'].values(),
        res_gauss['N'].values(),
        res_expon['N'].values(),
        solutions.values(),
        influxes
    )):
        rmse_gauss['K'][_influx] = rrmse(model_gauss(_solution.x, _rgk.x), _solution.Km[:,-1])
        rmse_gauss['N'][_influx] = rrmse(model_gauss(_solution.x, _rgn.x), _solution.Nm[:,-1])
        rmse_expon['K'][_influx] = rrmse(model_expon(_solution.x, _rek.x), _solution.Km[:,-1])
        rmse_expon['N'][_influx] = rrmse(model_expon(_solution.x, _ren.x), _solution.Nm[:,-1])


    return rmse_expon, rmse_gauss


@app.cell
def _(
    influxes,
    model_expon,
    model_gauss,
    np,
    plt,
    res_expon,
    res_gauss,
    rmse_expon,
    rmse_gauss,
    solutions,
):
    _influx = influxes[0]

    solution = solutions[_influx]
    x_plot = np.linspace(solution.x.min(), solution.x.max(), 100)
    rgk, rgn = res_gauss['K'][_influx], res_gauss['N'][_influx]
    rek, ren = res_expon['K'][_influx], res_expon['N'][_influx]


    fig, axs = plt.subplots(1,3, figsize=(12,3), dpi=300)

    ax = axs[0]
    ax.plot(solution.x, solution.Km[:,-1], color='black', lw=2, label='Simulation')
    ax.plot(
        x_plot, model_gauss(x_plot, rgk.x), color='tab:green', lw=3,
        linestyle=(0,(8,2)),
        label=f'Gauss fit (RRMSE = {rmse_gauss['K'][_influx]:.2f})'
    )
    ax.plot(
        x_plot, model_expon(x_plot, rek.x), lw=1, linestyle=(0,(2,2)), color='tab:green',
        label=f'Expon fit (RRMSE = {rmse_expon['K'][_influx]:.2f})'
    )
    ax.legend(frameon=False, loc='upper left')
    ax.set_ylim(-0.05,0.9)

    ax.set_xlabel(r'Position relative to induction $[\mu m]$')
    ax.set_ylabel(r'Activated CaMKII $[\mu M]$')
    ax.set_title('CaMKII activation profile\n'+rf'(${_influx:.0f}\, \mu M \, Ca^{{2+}}$per uncaging pulse)')

    ax = axs[1]
    ax.plot(solution.x, solution.Nm[:,-1], color='black', lw=2, label='Simulation')
    ax.plot(
        x_plot, model_gauss(x_plot, rgn.x), color='tab:purple', lw=3,
        linestyle=(0,(8,2)),
        label=f'Gauss fit (RRMSE = {rmse_gauss['N'][_influx]:.2f})'
    )
    ax.plot(
        x_plot, model_expon(x_plot, ren.x), lw=1, linestyle=(0,(2,2)), color='tab:purple',
        label=f'Expon fit (RRMSE = {rmse_expon['N'][_influx]:.2f})'
    )
    ax.legend(frameon=False, loc='upper left')
    ax.set_ylim(-0.4,11)

    ax.set_xlabel(r'Position relative to induction $[\mu m]$')
    ax.set_ylabel(r'Activated CaN $[\mu M]$')
    ax.set_title('CaN activation profile\n'+rf'(${_influx:.0f}\, \mu M \, Ca^{{2+}}$per uncaging pulse)')


    ax = axs[2]
    ax.scatter(
        rmse_gauss['K'].values(), rmse_expon['K'].values(), s=50, marker='o',
        cmap='Greens', c=solutions.keys(), facecolor='white',
        ec='tab:green',
        label='fit to CaMKII'
    )
    points = ax.scatter(
        rmse_gauss['N'].values(), rmse_expon['N'].values(), s=50, marker='o',
        cmap='Purples', c=solutions.keys(), facecolor='white',
        label='Fit to CaN', ec='tab:purple'
    )
    ax.plot((0.04,0.15), (0.04,0.15), c='black', lw=1, linestyle=(0,(10,4)))
    plt.colorbar(points)

    ax.legend(frameon=False, loc='lower right')
    ax.set_xlabel('RRMSE gaussian fit')
    ax.set_ylabel('RRMSE exponential fit')
    ax.set_title('Gauss/expon goodness\nof fit comparison')

    fig.subplots_adjust(wspace=0.3)
    plt.show()
    return (solution,)


@app.cell
def _(du, np, orig_params, replace, tqdm):
    solutions_spread = {}

    fs = np.arange(1,10, 20, 50, 100)

    for _f in tqdm(fs):
        _params = replace(
            orig_params,
            D_K = orig_params.D_K * _f,
            D_N = orig_params.D_N * _f,
        )
        _model = du.Model(_params)

        _solution = _model.simulate()

        solutions_spread[_f] = _solution
    return (solutions_spread,)


@app.cell
def _(np, plt, solution, solutions_spread, trapezoid):
    ref_K = trapezoid(solutions_spread[1].Km, x=solutions_spread[1].t, axis=1)
    ref_N = trapezoid(solutions_spread[1].Nm, x=solutions_spread[1].t, axis=1)

    colors = {
        'K' : plt.cm.Greens(np.linspace(0,1,5)),
        'N' : plt.cm.Purples(np.linspace(0,1,5)),
    }


    _fig, _axs = plt.subplots(1,2,figsize=(8,3), sharey=True, dpi=300)

    for _idx, (_f, _ss) in enumerate(solutions_spread.items()):
        if _idx == 0: continue
        
        _K = trapezoid(solutions_spread[_f].Km, x=solutions_spread[_f].t, axis=1)
        _N = trapezoid(solutions_spread[_f].Nm, x=solutions_spread[_f].t, axis=1)
    
        _axs[0].plot(solution.x, (_K - ref_K)/ref_K*100, color=colors['K'][_idx])
        _axs[1].plot(solution.x, (_N - ref_N)/ref_N*100, color=colors['N'][_idx], label=_f)

    _axs[0].set_ylabel('Relative integrated CaMKII\nchange [%]')
    _axs[1].set_ylabel('Relative integrated CaN\nchange [%]')
    _axs[1].legend(frameon=False, loc=(1,0), title=r'$D\,/\,D_0$')
    _axs[0].set_title('Change in integrated catalytic activity in function of their diffusivity', ha='left', x=0.1, y=1.05)

    for _ax in _axs:
        _ax.set_xlabel(r'Position relative to induction $[\mu m]$')
        _ax.axhline(y=0, c='black', lw=0.5, linestyle=(0,(10,2)))

    plt.subplots_adjust(wspace=0.3)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
