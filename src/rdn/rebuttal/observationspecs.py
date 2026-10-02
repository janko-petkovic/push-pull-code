from typing import NamedTuple
import jax 
import jax.numpy as jnp
from rdn.rebuttal.parameters import Experiment


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


def compute_observation_specs(parameters, dendrite_indexes, spine_indexes):
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
