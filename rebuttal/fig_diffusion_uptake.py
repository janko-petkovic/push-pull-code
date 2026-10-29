import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Uptake and conversion of diffusive resource

    We suppose a calcium injection occurs at a given spine located at $x_i$.
    Let's try to see what happens to the steady state distribution of the following system:
    $$
    \begin{aligned}
        &k_t = v_k \, m \, \left( 1 - \frac{k}{k_{tot}} \right)\\[6pt]
        &n_t = v_n \, m \, \left( 1 - \frac{n}{n_{tot}} \right)\\[6pt]
        &m_t = v_m \, c \, \left( 1 - \frac{m + k + n}{m_{tot}} \right) - k_t - n_t\\[6pt]
        &c_t = D_c \, c_{xx} - v_m \, c \, \left( 1 - \frac{m+k+n}{m_{tot}} \right)
    \end{aligned}
    $$

    With no flux boundary conditions for c.

    Remember that all the costants regarding spine quantities are piecewise defined functions, i.e. they are different from zero only at spine locations.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    # plt.style.use('rdn.plotstyles.presentation')

    from scipy.optimize import curve_fit, least_squares

    return least_squares, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Helper functions
    """)
    return


@app.cell
def _(np):
    class Problem():
        def __init__(self, **params):
            self.par_dict = self._setup_parameter_dict(**params)


        def _setup_parameter_dict(self, **params) -> dict:
            '''
            Quick but unsafe way of coding parameters, the user has to remember
            to give everythin that is needed.
            '''

            v_ks = np.zeros(params['dendrite_length'])
            v_ks[1::params['inter_spine_distance']] = params['v_k']

            v_ns = np.zeros(params['dendrite_length'])
            v_ns[1::params['inter_spine_distance']] = params['v_n']

            v_ms = np.zeros(params['dendrite_length'])
            v_ms[1::params['inter_spine_distance']] = params['v_m']

            # Account for the time units of measure
            for k, v in params.items():
                if ('v_' in k) or ('tau_' in k):
                    print(k)
                    params[k] = v * params['dt']

            return dict(v_ks=v_ks, v_ns=v_ns, v_ms=v_ms, **params)


        def _DKDT(self, m, k):        
            return self.par_dict['v_ks'] * (1-k/self.par_dict['k_tot']) * m 


        def _DNDT(self, m, n):
            return self.par_dict['v_ns'] * (1-n/self.par_dict['n_tot']) * m 


        def _DMDT_production(self, c, m, k, n):
            return self.par_dict['v_ms'] * (1-(k+n+m)/self.par_dict['m_tot']) * c 


        def _DY(self, c, m, k, n, dt):
            '''
            Main euler-1 integration stepper
            '''

            dc = np.zeros_like(c)
            dm = np.zeros_like(m)
            dk = np.zeros_like(k)
            dn = np.zeros_like(n)


            dkdt = self._DKDT(m, k)
            dk += dt * dkdt

            dndt = self._DNDT(m, n)
            dn += dt * dndt

            dmdt_production = self._DMDT_production(c, m, k, n)


            dm += dt * (dmdt_production) - dk - dn


            dc[0] = dt * (self.par_dict['diff_c'] * (-c[0] + c[1]) - dmdt_production[0])
            dc[-1] = dt * (self.par_dict['diff_c'] * (-c[-1] + c[-2]) - dmdt_production[-1])

            dc[1:-1] = dt * (self.par_dict['diff_c'] * np.diff(c, 2) - dmdt_production[1:-1])


            return dc, dm, dk, dn



        def solve(self):
            '''
            Returns a result with the computed time series
            '''

            dt = self.par_dict['dt']
            t_steps = int(self.par_dict['t_integration']/dt)        

            # Initial conditions
            cs = np.zeros(self.par_dict['dendrite_length'])
            ms = np.zeros(self.par_dict['dendrite_length'])
            ks = np.zeros(self.par_dict['dendrite_length'])
            ns = np.zeros(self.par_dict['dendrite_length'])

            # Apply the stimulus
            cs[self.par_dict['idx_stimulus']] = self.par_dict['c_stimulus']

            # Initialize the time arrays
            cs_t = [cs.copy()]
            ms_t = [ms.copy()]
            ks_t = [ks.copy()]
            ns_t = [ns.copy()]
    
            for _ in range(t_steps):

                dc, dm, dk, dn = self._DY(cs_t[-1], ms_t[-1], ks_t[-1], ns_t[-1], dt)

                cs_t.append(cs_t[-1] + dc)
                ms_t.append(ms_t[-1] + dm)
                ks_t.append(ks_t[-1] + dk)
                ns_t.append(ns_t[-1] + dn)

            return Result(cs_t=cs_t, ms_t=ms_t, ks_t=ks_t, ns_t=ns_t)


    class Result():
        def __init__(self, **solution):

            self.solution = solution

        def __str__(self):
            return self.solution.__str__()

    return (Problem,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simulation
    """)
    return


@app.cell
def _(Problem):
    problem = Problem(

        # Geometry
        dendrite_length = 100,
        inter_spine_distance = 1,
        idx_stimulus = 50,

        # Calcios in
        c_stimulus = 3000,

        # CamKII
        v_k = 1,

        # Calcineuro
        v_n = 10,
        tau_n = 1e-3,

        # CaM
        v_m = 1,

        k_tot = 600.,
        n_tot = 100.,
        m_tot = 400.,

        diff_c = 200,

        # integration step [s] - to convert constants
        dt = 1e-3,

        # integration time [s]
        t_integration = 10
    )


    # Assume dt is expressed in s
    result = problem.solve()
    return (result,)


@app.cell
def _(np, plt, result):
    colors = ["", "", 'tab:green', 'violet']

    for (k, v), c in zip(result.solution.items(), colors):
        print(k)
        if 'k' in k or 'n' in k:
            v_inf = v[-1]

            x = np.where(v_inf > 0)
            y = v_inf[x]

            plt.plot(x[0],y, label=k, color=c)

    # plt.ylim(0,2000)
    plt.legend()
    return


@app.cell
def _(least_squares, np, plt, result):
    def plot_fit(species, ax, **kwargs):
        ks_inf = result.solution[species][-1]
        x = np.where(ks_inf > 0)
        y = ks_inf[x]
        x_plot = x[0]

        def cauchy(x, mu, a, b):
            return 1 / a * 1 / (1 + (x - mu) ** 2 / b ** 2)

        def gauss(x, sigma, scale):
            return scale * np.exp(-((x - 250) / sigma) ** 2)

        def res(p, x, y):
            return y - gauss(x, *p)
        fitres = least_squares(res, [1, 1], args=(x[0], y), loss='cauchy')
        ax.scatter(x_plot[::2], y[::2], c='tab:gray', marker='x', s=100, alpha=0.3, label='simulation')
        ax.plot(x_plot, gauss(x_plot, *fitres.x), **kwargs, label='gauss fit')
        plt.legend(fontsize=15)
    _fig = plt.figure(figsize=(8, 6), dpi=300)  # popt, pcov = curve_fit(res, x[0], y)
    gs = _fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.1)
    ax_up = _fig.add_subplot(gs[0])
    plot_fit('ks_t', ax_up, color='forestgreen', linewidth=4)
    ax_up.set_xticklabels('')
    ax_up.set_ylabel('K [A.U.]')
    _ax = _fig.add_subplot(gs[1], sharex=ax_up)
    plot_fit('ns_t', _ax, color='mediumvioletred', linewidth=4)
    _ax.set_xlabel('Position along dendrite [um]')
    _ax.set_ylabel('N [A.U.]')
    return


if __name__ == "__main__":
    app.run()
