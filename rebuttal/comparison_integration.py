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
    plt.style.use('default')


    return Path, plt, run_experiment, viz


@app.cell
def _(Path, experiment_diff, experiment_qss, plt, run_experiment, viz):
    def model_comparison():
        PATH_TO_PAR_FOLDER = Path(__file__).parent / "parameters/"
        PATH_TO_SAVE_FOLDER = Path(__file__).parent / "output/"
    
        # differential = PATH_TO_PAR_FOLDER / '3_stim_differential.toml'
        # qss = PATH_TO_PAR_FOLDER / '3_stim_qss.toml'

        # experiment_diff = run_experiment(
        #         differential, PATH_TO_SAVE_FOLDER, force_new_simulation=False)
        # experiment_qss = run_experiment(
        #         qss, PATH_TO_SAVE_FOLDER, force_new_simulation=False)

        diff_experiments = []
        qss_experiments = []

        for file in PATH_TO_PAR_FOLDER.iterdir():
            path_to_parameters = PATH_TO_PAR_FOLDER / file
            experiment = run_experiment(path_to_parameters, PATH_TO_SAVE_FOLDER)

            if "qss" in file: 
                qss_experiments.append(experiment)
            else:
                diff_experiments.append(experiment)

        t_idxs = (2,10,20,30,)
        x_view = [41,61]
        y_lims = dict(ps=(0.5, 2.5), ksns=(0, 20), ud=(0.9, 1.1))

        fig = plt.figure(figsize=(12,1), dpi=200)

        viz.plot_comparison(
            fig,
            (experiment_diff, experiment_qss,),
            t_idxs,
            x_view,
            summary='medianiq',
            # y_lims=y_lims,
            keys=('ps',),
        )


    def _main():
        model_comparison()
        plt.show()

    _main()
    return


if __name__ == "__main__":
    app.run()
