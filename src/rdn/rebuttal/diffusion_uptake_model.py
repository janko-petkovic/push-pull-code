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

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Params:
    # geometry / numerics
    length: float = 20.0        # um, domain [-L, L]
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


@dataclass
class DiffusionUptakeModel:
    '''the dendritic system, we take out the spine to simplify things,
    supposing that the overall behaviour should be just a scaled version

    quantities are: c, m, k, n
    start with simple problem
    '''

    p: Params = field(default_factory=Params)
    
    def __post_init__(self):
        p = self.p
        self.N = int(p.length / p.dx)
        N = self.N

        # We gotta build the laplacian tho
        main_diag = -2*np.ones(N)
        main_diag[0] = main_diag[-1] = 0
        self.laplacian = sparse.diags_array(
            (np.ones(N-1), main_diag, np.ones(N-1)),
            offsets=(-1,0,1),
        )
        
        self.jac_sparsity = None


    def RHS(self, t, y):
        '''y shape 4*N'''
        p, N = self.p, self.N
        c, m, k, n = y[:N], y[N:2*N], y[2*N:3*N], y[3*N:]

        dc = self.laplacian @ c

        return np.concatenate(
            (dc, np.zeros(3*N))
        )


    
    def simulate(self, t_end):
        y0 = np.ones(4*self.N)
        res = solve_ivp(
            self.RHS,
            t_span=(0,t_end),
            y0=y0,
        )

        breakpoint()
        

        




        

        




