from pathlib import Path
from rdn.rebuttal.experiment import run_experiment
from rdn.rebuttal import viz
import matplotlib.pyplot as plt
import jax 
jax.config.update('jax_enable_x64', True)


def test_bartol():
    PATH_TO_PAR_FOLDER = Path(__file__).parent / "parameters/"
    PATH_TO_SAVE_FOLDER = Path(__file__).parent / "output/"
    differential = PATH_TO_PAR_FOLDER / '2_bartol.toml'

    experiment_diff = run_experiment(
            differential, PATH_TO_SAVE_FOLDER, force_new_simulation=False)


    t_idxs = (2,10,20,30,)
    x_view = [41,61]
    y_lims = dict(ps=(0.5, 2.5),ksns=(0, 30),ud=(0.9, 1.1))

    # fig = plt.figure(figsize=(12,3), dpi=100)
    # viz.plot_snapshot(
    #     fig,
    #     experiment_diff,
    #     180,
    #     x_view = x_view,
    #     summary="medianiq",
    #     relative=True,
    #     y_lims = y_lims,
    # )
    #
    fig = plt.figure(figsize=(7,3), dpi=100)
    viz.plot_bartol(
        fig,
        experiment_diff,
    )

    plt.show()

if __name__ == "__main__":
    test_bartol()
