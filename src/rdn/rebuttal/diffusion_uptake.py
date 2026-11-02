"""
Contains all the objects for the diffusion uptake figures.

Spatial activation profile of CaMKII and calcineurin (CaN) along a dendrite,
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
from tqdm import tqdm

@dataclass(frozen=True)
class Params:
    # geometry / numerics
    half_length: float = 10.0        # um, domain [-L, L]
    dx: float = 0.1           # um

    # calcium
    D_c: float = 13.0           # um^2/s, effective (Allbritton et al. 1992)
    tau_c: float = 0.05         # s, removal (pumps + buffers)
    kappa: float = 20.0         # endogenous buffer capacity (Sabatini 2002)

    # CaM
    m_tot: float = 40.0         # uM (Li et al. 2020)
    D_m: float = 20.0           # um^2/s (Schmidt et al. 2007)
    k_on_cm: float = 1000.0       # 1/s, Ca2+ loading of CaM at saturating Ca2+
    k_h: float = 1.0            # uM, half-activation of CaM (assumption)
    k_off_cm: float = 10.0        # 1/s, C-lobe Ca2+ off-rate (approximate)
    
    # CaMKII
    K_tot: float = 80.0          # uM (Li et al. 2020)
    D_K: float = 0.08            # um^2/s (Lu et al. 2014)
    D_Km: float = 0.009          # um^2/s (Lu et al. 2014)
    k_on_mk: float = 2.9         # 1/(uM s) (Li et al. 2020)
    k_off_mk: float = 1.1        # 1/s (Putkey & Waxham 1996)

    # CaN
    N_tot: float = 8.0           # uM (Li et al. 2020)
    D_N: float = 2.0             # um^2/s (estimate Swaminathan et al. 1997)
    D_Nm: float = 0.             # um^2/s (estimate Swaminathan et al. 1997)
    k_on_mn: float = 23.0        # 1/(uM s) (Li et al. 2020)
    k_off_mn: float = 0.075      # 1/s (Kd ~ 3.3 nM)

    # stimulus: n_pulses at freq, each an influx of duration pulse_dur
    n_pulses: int = 60
    freq: float = 0.5           # Hz
    pulse_dur: float = 0.01     # s
    pulse_width: float = 0.25   # um, Gaussian footprint (spine-head size)
    influx: float = 50.0        # uM/s at the centre during a pulse
    t_after: float = 10.0       # s of recording after the last pulse


@dataclass(frozen=True)
class Solution:
    t: np.array
    x: np.array
    c: np.array
    m: np.array
    mca: np.array
    K: np.array
    Km: np.array
    N: np.array
    Nm: np.array

    def fields(self):
        return ( self.c, self.m, self.K, self.Km, self.N, self.Nm)


@dataclass
class Model:
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
        )/self.p.dx**2
        
        # SPARSITY JACOBIAN
        # 1. first hte reactions of the species at the point of contact
        point = sparse.kron(
            np.ones((7,7)),         # big blocks = species
            sparse.eye_array(N)     # small cells = x coordinate
        )

        # 2. then the tridiagonal diffusion
        tri = sparse.diags_array(
            (1.,1.,1.,), offsets=(-1,0,1), shape=(N,N)
        )
        # 3. empty diffusion for n and k
        empty = sparse.csr_array((N,N))

        self.jac_sparsity = (
            point + sparse.block_diag([tri, tri, tri, tri, tri, tri, tri], format='csr')
        ).astype(bool)


    def RHS(self, t, y, stim_on,):
        '''y shape 4*N'''
        p, n_nodes = self.p, self.n_nodes
        c, m, mca, K, Km, N, Nm = [
            y[i*n_nodes:(i+1)*n_nodes] for i in range(7)
        ]

        # intermediates
        c_pos = np.maximum(c, 0.0) # in case c drops below 0 numerically
        h_c = c_pos**4 / (p.k_h**4 + c_pos**4) # as the even power would explode


        # stimulus c influx
        s = p.influx * self.c_profile if stim_on else 0.0

        #here go the equations
        Km_flux = p.k_on_mk * K * mca - p.k_off_mk * Km
        Nm_flux = p.k_on_mn * N * mca - p.k_off_mn * Nm

        dKm = p.D_Km * self.laplacian @ Km + Km_flux
        dNm = p.D_Nm * self.laplacian @ Nm + Nm_flux

        dK = p.D_K * self.laplacian @ K - Km_flux
        dN = p.D_N * self.laplacian @ N - Nm_flux

        mca_flux = p.k_on_cm * m * h_c - p.k_off_cm * mca

        dm = p.D_m * self.laplacian @ m - mca_flux
        dmca = p.D_m * self.laplacian @ mca + mca_flux - Km_flux - Nm_flux

        dc = p.D_c * self.laplacian @ c - 4/(1+p.kappa)*mca_flux - c/p.tau_c + s


        return np.concatenate((dc, dm, dmca, dK, dKm, dN, dNm))


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

        c0 = np.zeros(self.n_nodes)
        m0 = np.ones(self.n_nodes) * p.m_tot
        mca0 = np.zeros(self.n_nodes)
        K0 = np.ones(self.n_nodes) * p.K_tot
        Km0 = np.zeros(self.n_nodes)
        N0 = np.ones(self.n_nodes) * p.N_tot
        Nm0 = np.zeros(self.n_nodes)

        y0 = np.concatenate((c0, m0, mca0, K0, Km0, N0, Nm0))

        solution_t = []
        solution_y = dict(c = [], m = [], mca = [], K = [], Km = [], N = [],
                          Nm = [])

        for (ta, tb, stim_on) in tqdm(segments):
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
                solution_y.keys(), sol.y.reshape(7,-1,sol.y.shape[-1])
            ):
                solution_y[key].append(field)

            y0 = sol.y[:,-1]

        solution_t = np.concat(solution_t)
        for k, v in solution_y.items():
            solution_y[k] = np.concatenate(v, axis=1)
        

        return Solution(solution_t, self.x, *solution_y.values())

