from time import sleep
from dataclasses import dataclass, field
from pathlib import Path

from rdn.rebuttal.experiment import run_experiment
import rdn.rebuttal.diffusion_uptake as du
from rdn.rebuttal.utils import Timer

import matplotlib.pyplot as plt


def test_diffusion_uptake_model():
    m = du.Model(du.Params)

    with Timer(m.simulate) as f:
        res = f()

    fig, axs = plt.subplots(6,1,figsize=(4,6), sharex=True)

    for ax, fres in zip(axs, res.fields()):
        ax.plot(res.x, fres[:,-1])

    plt.show()
