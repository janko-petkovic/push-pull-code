import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from rdn.rebuttal.experiment import run_experiment
    from rdn.rebuttal import viz
    import matplotlib.pyplot as plt
    import jax 
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_debug_nans', True)
    plt.style.use('default')
    return Path, plt, run_experiment, viz


@app.cell
def _(Path, plt, run_experiment, viz):
    def model_comparison():
        root = Path(__file__).parent
    
        PATH_TO_PARAMETERS =  root / "parameters/" / "bartol" / "2_stim_differential.toml"
        PATH_TO_SAVE_FOLDER = root / "output/"

        experiment_diff = run_experiment(PATH_TO_PARAMETERS, PATH_TO_SAVE_FOLDER)

        fig = plt.figure(figsize=(7,3), dpi=100)
        viz.plot_bartol(
            fig,
            experiment_diff,
        )


    data = model_comparison()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
