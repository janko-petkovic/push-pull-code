import numpy as np
from pathlib import Path
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from rdn.rebuttal.experiment import run_experiment
from rdn.rebuttal import viz 



jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)


def run_simulation():
    PATH_TO_PAR_FOLDER = Path(__file__).parent / "parameters/"
    PATH_TO_SAVE_FOLDER = Path(__file__).parent / "output/"

    experiments = []

    for file in PATH_TO_PAR_FOLDER.iterdir():
        print(file)
        path_to_parameters = PATH_TO_PAR_FOLDER / file
        experiment = run_experiment(path_to_parameters, PATH_TO_SAVE_FOLDER)
        experiments.append(experiment)

    t_idxs = (2,10,20,30,)
    x_view = [41,61]
    y_lims = dict(ps=(0.5, 2.5),ksns=(0, 20),ud=(0.9, 1.1))


    for experiment in experiments:
        fig = plt.figure(figsize=(12,3), dpi=100)
        subfigs = fig.subfigures(1,4, wspace=0., hspace=0)

        for fig, t_idx in zip(subfigs.flatten(), t_idxs):
            fig.subplots_adjust(left=0.11, bottom=0.07, top=0.9, right=0.95)
            viz.plot_snapshot(fig, experiment, t_idx, x_view=x_view,
                              summary='medianiq', y_lims=y_lims)


    plt.show()


if __name__ == "__main__":
    run_simulation()
    # test_parameters()
