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
from time import time

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

@dataclass(frozen=True)
class Params:
    ...

@dataclass(frozen=True)
class DiffusionUptakeResult:
    t: np.array
    c: np.array
    m: np.array
    K: np.array
    N: np.array

    def fields(self):
        return ( self.c, self.m, self.K, self.N)

@dataclass
class DiffusionUptakeModel:
    '''the dendritic system, we take out the spine to simplify things,
    supposing that the overall behaviour should be just a scaled version

    quantities are: c, m, k, n
    start with simple problem
    '''
    p: Params = field(default_factory=Params)
    
    def __post_init__(self):
        self.x = np.arange(
            -self.p.half_length, self.p.half_length + self.p.dx / 2, self.p.dx
        )
        self.n_nodes = self.x.size

        self.c_profile = np.exp(-self.x**2/(2 * self.p.pulse_width**2))

        
        # LAPLACIAN
        N = self.n_nodes
        main_diag = -2*np.ones(N)
        main_diag[0] = main_diag[-1] = -1
        self.laplacian = sparse.diags_array(
            (np.ones(N-1), main_diag, np.ones(N-1)),
            offsets=(-1,0,1),
        )
        
        # SPARSITY JACOBIAN
        # 1. first hte reactions of the species at the point of contact
        point = sparse.kron(
            np.ones((4,4)),         # big blocks = species
            sparse.eye_array(N)     # small cells = x coordinate
        )

        # 2. then the tridiagonal diffusion
        tri = sparse.diags_array(
            (1.,1.,1.,), offsets=(-1,0,1), shape=(N,N)
        )
        # 3. empty diffusion for n and k
        empty = sparse.csr_array((N,N))

        self.jac_sparsity = (
            point + sparse.block_diag([tri, tri, empty, empty], format='csr')
        ).astype(bool)


    def RHS(self, t, y, stim_on,):
        '''y shape 4*N'''
        p, n_nodes = self.p, self.n_nodes
        c, m, K, N = (
            y[:n_nodes], y[n_nodes:2*n_nodes], y[2*n_nodes:3*n_nodes],
            y[3*n_nodes:]
        )

        # intermediates
        m_free = p.m_tot - m - K - N
        c_pos = np.maximum(c, 0.0) # in case c drops below 0 numerically
        h_c = c_pos**4 / (p.k_h**4 + c_pos**4) # as the even power would explode
        mc_flux = p.k_on_cm * m_free * h_c - p.k_off_cm * m

        # stimulus c influx
        s = p.influx * self.c_profile if stim_on else 0.0

        #here go the equations
        dK = p.k_on_mk * (p.K_tot - K) * m - p.k_off_mk * K
        dN = p.k_on_mn * (p.N_tot - N) * m - p.k_off_mn * N

        dc = p.D_c * self.laplacian @ c - 4/(1+p.kappa)*mc_flux - c/p.tau_c + s
        dm = p.D_m * self.laplacian @ m + mc_flux - dK - dN

        return np.concatenate((dc, dm, dK, dN))


    def simulate(self):
        p = self.p
        period = 1/p.freq
        segments = []

        for i in range(p.n_pulses):
            t0 = i * period
            # The bool is the presence/abscence of stimulation
            segments.append((t0, t0 + p.pulse_dur, True))
            segments.append((t0 + p.pulse_dur, t0 + period, False))
            
        t_end = p.n_pulses * period + p.t_after
        segments.append((p.n_pulses * period, t_end, False))

        y0 = np.zeros(4*self.n_nodes)

        solution_t = []
        solution_y = dict(c = [], m = [], K = [], N = [],)

        for (ta, tb, stim_on) in segments:
            sol = solve_ivp(
                self.RHS,
                t_span=(ta,tb),
                args = (stim_on,),
                y0=y0,
                rtol=1e-6,
                atol=1e-9,
                jac_sparsity=self.jac_sparsity,
                method='BDF',
                # max_step=1e-2,
            )

            if not sol.success:
                raise RuntimeError(sol.message)

            solution_t.append(sol.t)
            for key, field in zip(
                solution_y.keys(), sol.y.reshape(4,-1,sol.y.shape[-1])
            ):
                solution_y[key].append(field)

            y0 = sol.y[:,-1]

        solution_t = np.concat(solution_t)
        for k, v in solution_y.items():
            solution_y[k] = np.concatenate(v, axis=1)
        

        return DiffusionUptakeResult(solution_t, *solution_y.values())
