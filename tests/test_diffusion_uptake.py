from pathlib import Path
from rdn.rebuttal.experiment import run_experiment
from rdn.rebuttal.diffusion_uptake_model import DiffusionUptakeModel
import matplotlib.pyplot as plt


from dataclasses import dataclass, field

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

def test_diffusion_uptake_model():
    m = DiffusionUptakeModel(Params())
    m.simulate(1)


if __name__ == "__main__":
    test_qss_model()
