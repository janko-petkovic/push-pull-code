from pathlib import Path
from rdn.rebuttal.experiment import run_experiment
from rdn.rebuttal import viz
import matplotlib.pyplot as plt
import jax 
jax.config.update('jax_enable_x64', True)

def test_qss_model():
    PATH_TO_PAR_FOLDER = Path(__file__).parent / "parameters/"
    PATH_TO_SAVE_FOLDER = Path(__file__).parent / "output/"
    differential = PATH_TO_PAR_FOLDER / '5_stim_differential.toml'
    qss = PATH_TO_PAR_FOLDER / '5_stim_differential.toml'

    experiment_diff = run_experiment(
            differential, PATH_TO_SAVE_FOLDER, force_new_simulation=False)
    experiment_qss = run_experiment(
            qss, PATH_TO_SAVE_FOLDER, force_new_simulation=False)


    t_idxs = (2,10,20,30,)
    x_view = [41,61]
    y_lims = dict(ps=(0.5, 2.5),ksns=(0, 20),ud=(0.9, 1.1))

    fig = plt.figure(figsize=(12,3), dpi=100)
    subfigs = fig.subfigures(1,4, wspace=0., hspace=0)

    for fig, t_idx in zip(subfigs.flatten(), t_idxs):
        fig.subplots_adjust(left=0.11, bottom=0.07, top=0.9, right=0.95)
        viz.plot_snapshot(fig, experiment_diff, t_idx, x_view=x_view,
                          summary='medianiq', y_lims=y_lims)

    fig = plt.figure(figsize=(12,3), dpi=100)
    subfigs = fig.subfigures(1,4, wspace=0., hspace=0)

    for fig, t_idx in zip(subfigs.flatten(), t_idxs):
        fig.subplots_adjust(left=0.11, bottom=0.07, top=0.9, right=0.95)
        viz.plot_snapshot(fig, experiment_qss, t_idx, x_view=x_view,
                          summary='medianiq', y_lims=y_lims)

    plt.show()


if __name__ == "__main__":
    test_qss_model()
