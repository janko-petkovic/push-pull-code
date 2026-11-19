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
def _(Path, run_experiment):
    root = Path(__file__).parent
    stem = Path(__file__).stem
    PATH_TO_PAR_FOLDER_CONTROL =  root / "parameters/" / "control"

    # To generate the figures in the paper, just select the parameters that 
    # you want to visualize. Options are
    # - control
    # - no_diffusion
    # - fast_diffusion
    # - no_sigma
    # - wide_sigma
    PATH_TO_PAR_FOLDER_ALT =  root / "parameters/" / "no_diffusion"
    PATH_TO_SAVE_FOLDER = root / "output/"

    # I should parse the number of stimulations from inside the par file 
    # but for this time I will just parse it from the par string
    diff_experiments = dict(control={}, altered={})
    qss_experiments = dict(control={}, altered={})

    # Load the ablations
    for path_to_parameters in PATH_TO_PAR_FOLDER_ALT.iterdir():
        # Without diffusion
        experiment = run_experiment(path_to_parameters, PATH_TO_SAVE_FOLDER)

        if "qss" in path_to_parameters.stem: 
            qss_experiments['altered'][path_to_parameters.name[0]] = experiment
        else:
            diff_experiments['altered'][path_to_parameters.name[0]] = experiment


    # Load the controls
    for path_to_parameters in PATH_TO_PAR_FOLDER_CONTROL.iterdir():
        experiment = run_experiment(path_to_parameters, PATH_TO_SAVE_FOLDER)

        if "qss" in path_to_parameters.stem: 
            qss_experiments['control'][path_to_parameters.name[0]] = experiment
        else:
            diff_experiments['control'][path_to_parameters.name[0]] = experiment
    return diff_experiments, qss_experiments


@app.cell
def _(diff_experiments, plt, qss_experiments, viz):
    def model_comparison():
        t_idxs = (2,10,20,30)
        x_view = [41,61]
        y_lims = (-25,25)
        y_ticks = (-10,10)

        fig = plt.figure(figsize=(12,12), dpi=200)
        subfigs = fig.subfigures(4,1, hspace=0.1)

        for subfig, nss in zip(subfigs, [1,3,5,7]):
            viz.plot_comparison(
                subfig,
                (diff_experiments['control'][f'{nss}'],
                 qss_experiments['control'][f'{nss}'],),
                t_idxs,
                x_view,
                summary='medianiq',
                y_lims=y_lims,
                key='ps',
                compact=False,
                colors=('black', 'tab:blue'),
                y_ticks=y_ticks
            )

        return diff_experiments



    data = model_comparison()
    plt.show()
    return


@app.cell
def _(diff_experiments, plt, viz):
    def _main():
        t_idxs = (2,10,20,30)
        x_view = [41,61]
        y_lims = (-25,25)
        y_ticks = (-10,10)
    
        _fig = plt.figure(figsize=(12,1), dpi=300)
    
        
        viz.plot_snapshot(
            _fig,
            [diff_experiments['control'][nss] for nss in ['1','3','5','7']],
            keys=('ud',),
            t_idx = 2,
            x_view = x_view,
            ps_color='black',
            ud_color='black',
            y_lims={"ps":(0.5, 2), "ud":(0.99, 1.01)},
        )

        _fig.subplots_adjust(wspace=0.1)

    _main()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
