'''
A few notes on the function responsibilities

_run_session:
    1. istantiates the stepper and the observation/stimulation specs
    2. invokes the differential model simulator
    3. returns observation_specs, observed_y0, observed_yt, y being the observed
       quantities

_run_experiment:
    1. parses the experimental parameters
    2. runs the session repeats
    3. creates and returns the result object

run_experiment:
    Main interface of the simulator
    1. (safely) loads the parameters
    2. calls _run_experiment
    3. saves the result and returns it
'''

from __future__ import annotations
import hashlib
import pickle
from typing import NamedTuple
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import rdn.rebuttal.differential_model as diffm
from rdn.rebuttal.parameters import Parameters, Experiment



class ObservationSpecs(NamedTuple):
    """
    Attributes
    ----------
    obs_spine_arange: jax.Array
    obs_spine_indexes: jax.Array
    obs_spine_locations: jax.Array
    obs_dendrite_indexes: jax.Array
    obs_dendrite_locations: jax.Array
    obs_timesteps: jax.Array
    obs_times: jax.Array
    """

    obs_stim_locations: jax.Array
    obs_spine_arange: jax.Array
    obs_spine_indexes: jax.Array
    obs_spine_locations: jax.Array
    obs_dendrite_indexes: jax.Array
    obs_dendrite_locations: jax.Array
    obs_timesteps: jax.Array
    obs_times: jax.Array


class Dataset:
    """
    This is basically just a tuple of two dicts, all the methods are basically
    a shunt of the dict methods to the interface.

    Attributes
    ----------
    y0: dict
        Dict containing the initial values for the species
    yt: dict
        Dict containing the species's values at the observation times

    Totods: put a string representation at a certain point
    """

    y0: dict
    yt: dict

    def __init__(self, y0: dict, yt: dict):
        assert y0.keys() == yt.keys(), "Initial and observed y incompatible"
        self.y0 = y0
        self.yt = yt

    def keys(self):
        return self.y0.keys()

    def values(self):
        return (self.y0.values(), self.yt.values())

    def items(self):
        return [
            (k0, v0, vt)
            for (k0, v0), (_, vt) in zip(self.y0.items(), self.yt.items())
        ]

    def get_summary(
        self, key: str, summary: str = "medianiq", relative: bool = False
    ) -> tuple:
        """
        summary: medianiq or meansem
        """
        Y = self.yt[key]

        if relative:
            Y /= self.y0[key]

        if summary == "medianiq":
            l, m, h = jnp.quantile(Y, jnp.array((0.25, 0.5, 0.75)), axis=0)
        elif summary == "meansem":
            mean = jnp.mean(Y, axis=0)
            sem = jnp.std(Y, axis=0) / jnp.sqrt(len(Y))
            l, m, h = mean - sem, mean, mean + sem
        else:
            raise ValueError("summary value can be 'medianiq' or 'meansem'.")

        return l, m, h


class Result(NamedTuple):
    """Inspired to scipy's result object returned from solve_ivp."""

    key: jax._src.prng.PRNGKeyArray
    ts: jax.Array
    xs_stim: jax.Array
    xs_spine: jax.Array
    xs_dendrite: jax.Array
    dataset: Dataset

    @staticmethod
    def _load(filename):
        with open(filename, "rb") as f:
            experiment = pickle.load(f)
        return experiment

    @staticmethod
    def safe_load(
        parameters, path_to_save_folder
    ) -> tuple[str, Result | None]:
        """returns always the path to the file and optionally the experiment in
        question"""

        digest = hashlib.shake_128(f"{parameters}".encode()).hexdigest(5)
        filename = path_to_save_folder / f"simulation_{digest}.pkl"

        if filename.exists():
            print("Experiment already simulated. Loading existing result.")
            with open(filename, "rb") as f:
                experiment = pickle.load(f)
            return filename, experiment
        else:
            print("Experiment not present. You will have to simulate it.")
            return filename, None


    def get_X_and_mask_from_view(self, x_view: list):
        '''Returns the X and the mask to apply to the Y positions in order to
        plot the view in question '''

        dendrite_mask = (
            (self.xs_dendrite >= x_view[0])*(self.xs_dendrite < x_view[1])
        )
        dendrite_X = self.xs_dendrite[dendrite_mask]
        spine_mask = (self.xs_spine >= x_view[0])*(self.xs_spine < x_view[1])
        spine_X = self.xs_spine[spine_mask]

        return (
            dendrite_mask,
            dendrite_X,
            spine_mask,
            spine_X,
        )


    @staticmethod
    def save(experiment: Result, path_to_save_file: str) -> None:
        with open(path_to_save_file, "wb") as f:
            pickle.dump(experiment, f)


def _compute_observation_specs(parameters, dendrite_indexes, spine_indexes):
    """This function does a bit too much stuff to be a dangling function,
    better include it in the parameter class when we implement it"""

    dx = parameters.integration.dx
    dt = parameters.integration.dt
    obs_time_range = parameters.experiment.observation.time_range
    obs_time_factor = Experiment.time_unit_to_factor(
        parameters.experiment.time_unit
    )
    obs_times = jnp.arange(*obs_time_range)

    cpm = int(1 / dx)

    obs_timesteps = (obs_times * obs_time_factor / dt).astype(int)

    stim_indexes = jnp.array(
        parameters.experiment.uncaging_protocol.spine_locations
    )

    obs_spine_limits = parameters.experiment.observation.spine_limits
    obs_spine_arange = jnp.arange(obs_spine_limits[0], obs_spine_limits[1])

    obs_spine_indexes = spine_indexes[obs_spine_arange]
    obs_dendrite_indexes = dendrite_indexes[
        obs_spine_indexes[0] - cpm : obs_spine_indexes[-1] + cpm + 1
    ]

    obs_stim_locations = stim_indexes + 1
    obs_spine_locations = obs_spine_indexes * dx
    obs_dendrite_locations = obs_dendrite_indexes * dx

    return ObservationSpecs(
        obs_stim_locations,
        obs_spine_arange,
        obs_spine_indexes,
        obs_spine_locations,
        obs_dendrite_indexes,
        obs_dendrite_locations,
        obs_timesteps,
        obs_times,
    )


def _run_session(key, parameters) -> tuple:
    """
    The main simulation routine.

    Returns
    -------
    tuple:
        observation_specs
        observed_initial_amounts
        observed_amounts_at_obs_times
    """

    # Setting up
    ic_key, unc_key = jax.random.split(key)
    y0, dendrite_indexes, spine_indexes = (
        diffm.setup_dendrite_initial_conditions(ic_key, parameters)
    )
    unc_mask = diffm.setup_rdn_uncaging_mask(unc_key, parameters, y0)
    stepper = diffm.setup_rnd_integrator(
        parameters, spine_indexes, y0, unc_mask,
    )
    ospecs = _compute_observation_specs(
        parameters, dendrite_indexes, spine_indexes
    )

    ### INTEGRATION ###
    obs_yt = {k: [] for k in y0.keys()}
    for k, v in y0.items():
        if "d" in k:
            obs_yt[k].append(v[ospecs.obs_dendrite_indexes])
        else:
            obs_yt[k].append(v[ospecs.obs_spine_arange])

    yt = y0

    for low, high in zip(ospecs.obs_timesteps[:-1], ospecs.obs_timesteps[1:]):
        # Integration
        yt = jax.lax.fori_loop(low, high, stepper, yt)

        # Data extraction and creation of the
        for k, v in yt.items():
            if "d" in k:
                obs_yt[k].append(v[ospecs.obs_dendrite_indexes])
            else:
                obs_yt[k].append(v[ospecs.obs_spine_arange])
    ###################

    for k, v in obs_yt.items():
        obs_yt[k] = jnp.stack(v)

    obs_y0 = {}
    for k, v in y0.items():
        if "d" in k:
            obs_y0[k] = v[ospecs.obs_dendrite_indexes]
        else:
            obs_y0[k] = v[ospecs.obs_spine_arange]

    return (ospecs, obs_y0, obs_yt)


def _run_experiment(parameters) -> Result:
    """
    Actually computes the experimental run given the specs in the parameter
    file and the random key.
    """

    n_sessions = parameters.experiment.n_sessions
    key = jax.random.key(parameters.integration.seed)
    key, *subkeys = jax.random.split(key, n_sessions + 2)

    # Run one dry session to compile
    from time import time

    print("Executing dry run...", end="")
    timea = time()
    ospecs, obs_y0, _ = _run_session(subkeys[0], parameters)
    obs_y0["ks"].block_until_ready()
    timeb = time()
    session_runtime = timeb - timea
    print(f"session runtime (with compilation): {session_runtime:2f} s")

    v_run_session = jax.vmap(_run_session, (0, None))
    print("Running full experiment...", end="")
    timea = time()
    _, obs_y0, obs_yt = v_run_session(jnp.array(subkeys[1:]), parameters)
    obs_y0["ks"].block_until_ready()
    timeb = time()
    experiment_runtime = timeb - timea
    print(f"experiment runtime (with compilation): {experiment_runtime:2f} s")

    # Insert one dimension for compatibility with yt
    for k, v in obs_y0.items():
        obs_y0[k] = jnp.expand_dims(v, axis=1)
    dataset = Dataset(obs_y0, obs_yt)

    return Result(
        key,
        ospecs.obs_times,
        ospecs.obs_stim_locations,
        ospecs.obs_spine_locations,
        ospecs.obs_dendrite_locations,
        dataset,
    )


def run_experiment(
    path_to_parameters: Path,
    path_to_save_folder: Path, 
    model_type: str = 'differential',
) -> Result:
    """For the love of god, the paths are relative to the main function call,
    be certain of that

    model_type: differential or qss

    """

    parameters = Parameters.load(path_to_parameters)

    path_to_file, experiment = Result.safe_load(
        parameters, path_to_save_folder
    )

    if experiment is None:
        print("Simulating new experiment.")
        match model_type:
            case 'differential':
                result = _run_experiment(parameters)
            case 'qss':
                from rdn.rebuttal.qss_model import run_session_qss
                result = run_session_qss(parameters)
            case _:
                raise ValueError(
                    'Wrong model type. Possibilities are "differential" '
                    'and "qss"'
                )

        Result.save(experiment, path_to_file)

    return result
