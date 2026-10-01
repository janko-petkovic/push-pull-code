from pathlib import Path
import tomllib
from jax.random import key as rkey
from jax.lax import scan
import jax.numpy as jnp
import matplotlib.pyplot as plt

from rdn.rebuttal.differential_model import (
    setup_dendrite_initial_conditions,
    setup_rnd_integrator,
    setup_rdn_uncaging_time_array,
)
from rdn.rebuttal.experiment import _run_session, run_experiment


plt.rcParams["font.sans-serif"] = "Open Sans"


def test_setup_dendrite_initial_conditions(key, parameters):
    y0, dendritic_indexes, spine_indexes = setup_dendrite_initial_conditions(
        key, parameters
    )

    print(f"{y0["ud"].shape=},\t{y0['ud']}")
    print(f"{y0["us"].shape=},\t{y0['us']}")
    print(f"{y0["ks"].shape=},\t{y0['ks']}")
    print(f"{y0["ns"].shape=},\t{y0['ns']}")
    print(f"{y0["ps"].shape=},\t{y0['ps']}")
    print(f"{y0["spine_sizes"].shape=},\t{y0['spine_sizes']}")
    print(f"{spine_indexes=}")

    fig = plt.figure(figsize=(8, 4), dpi=100)
    axsd = fig.subplot_mosaic(
        """
        AACD
        BBEF
        """,
        gridspec_kw=dict(wspace=0.5, hspace=0.5),
    )
    axs = [ax for ax in axsd.values()]
    ax = axsd["A"]
    ax.plot(dendritic_indexes, y0["ud"], label="ud")
    ax.plot(spine_indexes, y0["us"], label="us")
    ax.set_xlim(-1, dendritic_indexes[-1] + 1)
    ax.set_xticks(spine_indexes)
    ax.set_xlabel("Position along dendrite [dx]")
    ax.set_ylabel("value")

    ax = axsd["B"]
    ax.plot(spine_indexes, y0["ks"], label="ks")
    ax.plot(spine_indexes, y0["ns"], label="ns")
    ax.plot(spine_indexes, y0["ps"], label="ps")
    ax.set_xlim(-1, dendritic_indexes[-1] + 1)
    ax.set_xticks(spine_indexes)
    ax.set_xlabel("Position along dendrite [dx]")
    ax.set_yscale("log")
    ax.set_ylabel("value")

    ax = axsd["C"]
    ax.scatter(y0["ks"], y0["ps"])
    ax.set_xlabel("ks")
    ax.set_ylabel("ps")

    ax = axsd["D"]
    ax.scatter(y0["ps"], y0["spine_sizes"])
    ax.set_xlabel("ps")
    ax.set_ylabel("sizes")

    ax = axsd["E"]
    ax.scatter(y0["ns"], y0["ps"])
    ax.set_xlabel("ns")
    ax.set_ylabel("ps")

    for ax in axs:
        ax.legend()


def test_setup_rnd_uncaging_time_array(parameters):
    unc_array = setup_rdn_uncaging_time_array(parameters)
    print(unc_array)


def test_rdn_integration(key, parameters):
    plot_stride = 60

    # Pipeline
    y0, dendrite_indexes, spine_indexes = setup_dendrite_initial_conditions(
        key, parameters
    )
    unc_time_array = setup_rdn_uncaging_time_array(parameters)

    stepper = setup_rnd_integrator(
        parameters, spine_indexes, y0, unc_time_array
    )
    _, yt = scan(
        stepper,
        y0,
        xs=(jnp.arange(parameters["integration"]["n_steps"]), unc_time_array),
    )

    # Plotting
    wr1 = 1 / parameters["integration"]["dx"]
    fig, axs = plt.subplots(
        1, 5, figsize=(15, 3), width_ratios=(wr1, 1, 1, 1, 1)
    )

    ax = axs[0]
    im = ax.imshow(
        yt["ud"][::plot_stride] / y0["ud"], cmap="Greys", vmin=0.5, vmax=1.5
    )

    ax = axs[1]
    im = ax.imshow(
        yt["us"][::plot_stride] / y0["us"], cmap="Greys", vmin=0.5, vmax=1.5
    )

    ax = axs[2]
    im = ax.imshow(
        yt["ks"][::plot_stride] / y0["ks"], cmap="Greens", vmin=1, vmax=5
    )

    ax = axs[3]
    im = ax.imshow(
        yt["ns"][::plot_stride] / y0["ns"], cmap="PuRd", vmin=1, vmax=5
    )

    ax = axs[4]
    im = ax.imshow(
        (yt["ps"] / y0["ps"])[::plot_stride],
        cmap="coolwarm_r",
        vmin=0.8,
        vmax=1.2,
    )


def test_rdn_run_session(key, parameters):
    t, x_spine, x_dend, y0, y = run_session(key, parameters)

    # This should be taken from the experiment
    ts_obs = parameters["experiment"]["observation"]["times"]

    fig, axs = plt.subplots(4, 1, figsize=(5, 5), dpi=100, sharex=True)

    ax = axs[0]
    for t_obs, yy in zip(ts_obs, y["ps"]):
        ax.plot(x_spine, yy / y0["ps"], "-o", label=t_obs)
    ax.set_ylabel("P")

    ax = axs[1]
    for t_obs, yy in zip(ts_obs, y["ks"]):
        ax.plot(x_spine, yy / y0["ks"], "-o", label=t_obs)
    ax.set_ylabel("K")

    ax = axs[2]
    for t_obs, yy in zip(ts_obs, y["ns"]):
        ax.plot(x_spine, yy / y0["ns"], "-o", label=t_obs)
    ax.set_ylabel("N")

    ax = axs[3]
    for t_obs, yy in zip(ts_obs, y["ud"]):
        ax.plot(x_dend, yy / y0["ud"], label=t_obs)
    ax.set_ylabel("U")

    ax.set_xticks(x_spine)
    ax.set_xlabel("Position along dendrite [um]")

    for ax in axs:
        ax.legend()

    for ax in axs[:-1]:
        ax.set_ylim(0.5, 10)


def test_rdn_run_experiment(key, parameters):
    experiment = run_experiment(key, parameters)

    fig, axs = plt.subplots(5, 1)

    for ax, k in zip(axs, experiment.dataset.y0.keys()):
        if "d" in k:
            X = experiment.xs_dendrite
        else:
            X = experiment.xs_spine
        Y = jnp.median(experiment.dataset.yt[k], axis=0)

        for t, y in zip(experiment.ts, Y):
            ax.plot(X, y, label=t)

        ax.set_xticks(experiment.xs_dendrite)
        ax.set_title(k)
        ax.legend()

    plt.show()


if __name__ == "__main__":
    parameter_path = (
        Path(__file__).parent / "parameters" / "parameters_1stim.toml"
    )
    run_experiment(parameter_path, ".")

    # key = rkey(parameters["integration"]["seed"])
    # test_setup_dendrite_initial_conditions(key, parameters)
    # test_setup_rnd_uncaging_time_array(parameters)
    # test_rdn_integration(key, parameters)
    # test_rdn_run_session(key, parameters)
    # test_rdn_run_experiment(key, parameters)

    plt.show()
