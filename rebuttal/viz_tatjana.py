from pathlib import Path
from rdn.experiment import run_experiment
from rdn import viz
import matplotlib.pyplot as plt
import jax

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = "Open Sans"


def run_simulation():
    PATH_TO_PAR_FOLDER = Path(__file__).parent / "parameters/"
    PATH_TO_SAVE_FOLDER = Path(__file__).parent / "output/"
    PATH_TO_PARAMETERS = PATH_TO_PAR_FOLDER / "test.toml"
    
    experiment = run_experiment(PATH_TO_PARAMETERS, PATH_TO_SAVE_FOLDER)

    t_idx = 20
    x_view = [0,101]

    fig, axs = plt.subplots(2,1,figsize=(12,3), dpi=100, sharex=True)
    fig.subplots_adjust(left=0.05, bottom=0.2, top=1, right=0.99)
    viz.plot_cool(axs[0], experiment, 'ps',  t_idx, x_view=x_view,
                  summary='medianiq', showstim=True,
                  label='Median spine size change')
    viz.plot_cool(axs[1], experiment, 'ud',  t_idx, x_view=x_view,
                  summary='medianiq', label='Median dendritic resource change')
    # viz.plot_field(ax, experiment, 'ud', x_view, relative=True)

    plt.savefig(f'figures/plasticity_{t_idx}.svg')
    plt.show()


if __name__ == "__main__":
    run_simulation()
    # test_parameters()
