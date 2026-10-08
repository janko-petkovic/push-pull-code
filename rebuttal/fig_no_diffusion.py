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
def _(Path, plt, run_experiment, viz):
    def model_comparison():
        root = Path(__file__).parent
        stem = Path(__file__).stem
        PATH_TO_PAR_FOLDER_NO =  root / "parameters/" / stem / "no_diffusion"
        PATH_TO_PAR_FOLDER_WITH =  root / "parameters/" / stem / "with_diffusion"
        PATH_TO_SAVE_FOLDER = root / "output/"

        # I should parse the number of stimulations from inside the par file 
        # but for this time I will just parse it from the par string
        diff_experiments = dict(with_diffusion={}, no_diffusion={})
        qss_experiments = dict(with_diffusion={}, no_diffusion={})

        for path_to_parameters in PATH_TO_PAR_FOLDER_NO.iterdir():
            # Without diffusion
            experiment = run_experiment(path_to_parameters, PATH_TO_SAVE_FOLDER)

            if "qss" in path_to_parameters.stem: 
                qss_experiments['no_diffusion'][path_to_parameters.name[0]] = experiment
            else:
                diff_experiments['no_diffusion'][path_to_parameters.name[0]] = experiment

        for path_to_parameters in PATH_TO_PAR_FOLDER_WITH.iterdir():
            experiment = run_experiment(path_to_parameters, PATH_TO_SAVE_FOLDER)

            if "qss" in path_to_parameters.stem: 
                qss_experiments['with_diffusion'][path_to_parameters.name[0]] = experiment
            else:
                diff_experiments['with_diffusion'][path_to_parameters.name[0]] = experiment

    
        t_idxs = (2,10,20,30)
        x_view = [41,61]
        y_lims = dict(ps=(0.5, 2.5), ksns=(0, 20), ud=(0.9, 1.1))

        fig = plt.figure(figsize=(12,12), dpi=200)
        subfigs = fig.subfigures(4,1, hspace=0.1)
    
        for subfig, nss in zip(subfigs, [1,3,5,7]):
            viz.plot_comparison(
                subfig,
                (diff_experiments['no_diffusion'][f'{nss}'],
                 diff_experiments['with_diffusion'][f'{nss}'],),
                t_idxs,
                x_view,
                summary='medianiq',
                y_lims=y_lims,
                key='ps',
                compact=False,
                colors=('black', 'purple'),
            )


    def _main():
        model_comparison()
        plt.show()

    _main()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
