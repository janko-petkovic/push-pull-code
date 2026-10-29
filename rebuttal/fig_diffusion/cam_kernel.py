"""Spatial activation profile of CaMKII and calcineurin (CaN) along a dendrite,
after repeated calcium influx at one location (x = 0).

Units: uM, s, um.

    c_t = D_c c_xx - c/tau_c - 4/(1+kappa) * (k_act a h(c) - koff_m m) + s(x,t)
    m_t = D_m m_xx + k_act a h(c) - koff_m m
          - kon_K m (K_tot - k) + koff_K k - kon_N m (N_tot - n) + koff_N n
    k_t = kon_K m (K_tot - k) - koff_K k
    n_t = kon_N m (N_tot - n) - koff_N n

c: free Ca2+ (D_c, tau_c and kappa lump the other buffers and the pumps)
m: free Ca4-CaM, diffusing; a = m_tot - m - k - n: apo-CaM (locally conserved)
k, n: CaM-bound (active) CaMKII and CaN, immobile
h(c) = c^4 / (K_h^4 + c^4): cooperative Ca2+ loading of CaM (4 Ca2+ per CaM)

Readout: time-integrated active CaMKII and CaN, int k dt and int n dt, i.e.
the accumulated activation that the push-pull kernel represents.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Params:
    # geometry / numerics
    half_length: float = 10.0   # um, domain [-L, L]
    dx: float = 0.05            # um
    # calcium
    D_c: float = 13.0           # um^2/s, effective (Allbritton et al. 1992)
    tau_c: float = 0.05         # s, removal (pumps + buffers)
    kappa: float = 20.0         # endogenous buffer capacity (Sabatini 2002)
    # CaM
    m_tot: float = 40.0         # uM (Li et al. 2020)
    D_m: float = 20.0           # um^2/s (Schmidt et al. 2007)
    k_act: float = 1000.0       # 1/s, Ca2+ loading of CaM at saturating Ca2+
    K_h: float = 1.0            # uM, half-activation of CaM (assumption)
    koff_m: float = 10.0        # 1/s, C-lobe Ca2+ off-rate (approximate)
    # CaMKII
    K_tot: float = 80.0         # uM (Li et al. 2020)
    kon_K: float = 2.9          # 1/(uM s) (Li et al. 2020)
    koff_K: float = 1.1         # 1/s (Putkey & Waxham 1996)
    # CaN
    N_tot: float = 8.0          # uM (Li et al. 2020)
    kon_N: float = 23.0         # 1/(uM s) (Li et al. 2020)
    koff_N: float = 0.075       # 1/s (Kd ~ 3.3 nM)
    # stimulus: n_pulses at freq, each an influx of duration pulse_dur
    n_pulses: int = 60
    freq: float = 0.5           # Hz
    pulse_dur: float = 0.01     # s
    pulse_width: float = 0.25   # um, Gaussian footprint (spine-head size)
    influx: float = 2000.0      # uM/s at the centre during a pulse
    t_after: float = 30.0       # s of recording after the last pulse


class Model:
    def __init__(self, p: Params):
        self.p = p
        self.x = np.arange(-p.half_length, p.half_length + p.dx / 2, p.dx)
        self.N = self.x.size
        N = self.N
        # Laplacian with no-flux boundaries
        main = -2 * np.ones(N); main[0] = main[-1] = -1
        self.lap = sparse.diags(
            [np.ones(N - 1), main, np.ones(N - 1)], [-1, 0, 1]
        ) / p.dx**2
        self.footprint = np.exp(-self.x**2 / (2 * p.pulse_width**2))

        # Jacobian sparsity: diffusion is tridiagonal, reactions are local
        local = sparse.kron(np.ones((4, 4)), sparse.eye(N))
        tri = sparse.diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(N, N))
        self.jac_sparsity = (
            local + sparse.block_diag([tri, tri, sparse.csr_matrix((N, N)),
                                       sparse.csr_matrix((N, N))])
        ).astype(bool)

    def rhs(self, t, y, stim_on):
        p, N = self.p, self.N
        c, m, k, n = y[:N], y[N:2*N], y[2*N:3*N], y[3*N:]
        c_pos = np.maximum(c, 0.0)
        a = p.m_tot - m - k - n
        h = c_pos**4 / (p.K_h**4 + c_pos**4)
        load = p.k_act * a * h - p.koff_m * m          # net CaM loading
        bind_K = p.kon_K * m * (p.K_tot - k) - p.koff_K * k
        bind_N = p.kon_N * m * (p.N_tot - n) - p.koff_N * n
        s = p.influx * self.footprint if stim_on else 0.0

        dc = p.D_c * (self.lap @ c) - c / p.tau_c - 4 / (1 + p.kappa) * load + s
        dm = p.D_m * (self.lap @ m) + load - bind_K - bind_N
        return np.concatenate([dc, dm, bind_K, bind_N])

    def run(self, max_step_off=0.05):
        """Integrate pulse by pulse; returns times, states, and int k dt,
        int n dt (trapezoidal over the dense output)."""
        p, N = self.p, self.N
        y = np.zeros(4 * N)
        period = 1 / p.freq
        segments = []
        for i in range(p.n_pulses):
            t0 = i * period
            segments.append((t0, t0 + p.pulse_dur, True))
            segments.append((t0 + p.pulse_dur, t0 + period, False))

        t_end = (p.n_pulses - 1) * period + period + p.t_after

        segments.append((p.n_pulses * period, t_end, False))

        int_k = np.zeros(N); int_n = np.zeros(N)
        peak_c = np.zeros(N)
        for (ta, tb, on) in segments:
            sol = solve_ivp(
                self.rhs, (ta, tb), y, args=(on,), method="BDF",
                jac_sparsity=self.jac_sparsity, rtol=1e-6, atol=1e-9,
                max_step=(p.pulse_dur / 5 if on else max_step_off),
            )
            if not sol.success:
                raise RuntimeError(sol.message)
            Y = sol.y
            int_k += np.trapezoid(Y[2*N:3*N], sol.t, axis=1)
            int_n += np.trapezoid(Y[3*N:], sol.t, axis=1)
            peak_c = np.maximum(peak_c, Y[:N].max(axis=1))
            y = Y[:, -1]

        return dict(x=self.x, int_k=int_k, int_n=int_n, peak_c=peak_c,
                    final=y)


# ------------------------------------------------------------------ fitting
def gauss(x, s, A):
    return A * np.exp(-(x / s) ** 2)            # same form as bell_kernel


def expo(x, lam, A):
    return A * np.exp(-np.abs(x) / lam)


def fit_shapes(x, y, rel_floor=1e-3):
    """Fit Gaussian and exponential (centred at 0) on the points above
    rel_floor * max. Returns widths and 1 - R^2 for both."""
    from scipy.optimize import curve_fit
    sel = y > rel_floor * y.max()
    out = {}
    for name, f in (("gauss", gauss), ("exp", expo)):
        popt, _ = curve_fit(f, x[sel], y[sel], p0=(0.5, y.max()),
                            bounds=([1e-3, 0], [np.inf, np.inf]),
                            maxfev=20000)
        res = y[sel] - f(x[sel], *popt)
        out[name] = dict(width=popt[0], amp=popt[1],
                         one_minus_r2=np.sum(res**2)
                         / np.sum((y[sel] - y[sel].mean())**2))
    return out


def neighbour_weights(x, y, distances=(1.0, 2.0)):
    """Activation at the neighbouring spines relative to the stimulated one."""
    y0 = np.interp(0.0, x, y)
    return {d: 0.5 * (np.interp(d, x, y) + np.interp(-d, x, y)) / y0
            for d in distances}
