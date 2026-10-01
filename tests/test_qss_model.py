from pathlib import Path
from rdn.rebuttal.qss_model import run_session_qss
from rdn.rebuttal.experiment import run_experiment
from rdn.rebuttal import viz
import jax
import matplotlib.pyplot as plt

def test_qss_model():
    key = jax.random.key(2026)

    PATH_TO_PAR_FOLDER = Path(__file__).parent / "parameters/"
    PATH_TO_SAVE_FOLDER = Path(__file__).parent / "output/"
    path_to_parameters = PATH_TO_PAR_FOLDER / '1_stim.toml'

    experiment = run_experiment(path_to_parameters, PATH_TO_SAVE_FOLDER, 'qss')

    breakpoint()


    t_idxs = (2,10,20,30,)
    x_view = [41,61]
    y_lims = dict(ps=(0.5, 2.5),ksns=(0, 20),ud=(0.9, 1.1))

    fig = plt.figure(figsize=(12,3), dpi=100)
    subfigs = fig.subfigures(1,4, wspace=0., hspace=0)

    for fig, t_idx in zip(subfigs.flatten(), t_idxs):
        fig.subplots_adjust(left=0.11, bottom=0.07, top=0.9, right=0.95)
        viz.plot_snapshot(fig, experiment, t_idx, x_view=x_view,
                          summary='medianiq', y_lims=y_lims)

    plt.show()
