import jax
import jax.numpy as jnp
from rdn.rebuttal.parameters import Experiment
from rdn.rebuttal.observationspecs import compute_observation_specs
import rdn.rebuttal.differential_model as diffm
from rdn.rebuttal.differential_model import bell_kernel

def run_session_qss(key, parameters) -> tuple:
    '''remember that spines are hardcoded to be spaced by 1um from each
    other.
    each yy needs to have shape (n_sessions, n_times, n_spines)
    '''
    
    OMEGA = 200

    # define the local parameters
    n_spines = parameters.dendrite.n_spines
    unc_locations = jnp.array(
        parameters.experiment.uncaging_protocol.spine_locations
    )
    obs_time_factor = Experiment.time_unit_to_factor(
        parameters.experiment.time_unit
    )

    sigma_K = parameters.species.kinases.sigma
    sigma_N = parameters.species.phosphatases.sigma
    tau_K = parameters.species.kinases.tau / obs_time_factor
    tau_N = parameters.species.phosphatases.tau / obs_time_factor
    Ks = parameters.species.kinases.delta_stim
    Ns = parameters.species.phosphatases.delta_stim

    # Generate initial conditions
    ic_key, unc_key = jax.random.split(key)

    y0, dendrite_indexes, spine_indexes = (
        diffm.setup_dendrite_initial_conditions(ic_key, parameters)
    )
    
    ospecs = compute_observation_specs(
        parameters, dendrite_indexes, spine_indexes
    )

    # Basal catalysts
    K_basal_x = y0['ks']
    N_basal_x = y0['ns']

    # Compute the contributions
    v_bell_kernel = jax.vmap(bell_kernel, (None, 0, None))

    Ks0_x = v_bell_kernel(
        jnp.arange(n_spines),
        unc_locations,
        sigma_K,
    ).sum(axis=0) * Ks*100

    Ns0_x = v_bell_kernel(
        jnp.arange(n_spines),
        unc_locations,
        sigma_N,
    ).sum(axis=0) * Ns*100

    K_tx = (
        Ks0_x[None,:] * jnp.exp(-ospecs.obs_times/tau_K)[:, None] 
        + K_basal_x[None, :]
    )

    N_tx = (
        Ns0_x[None,:] * jnp.exp(-ospecs.obs_times/tau_N)[:, None] 
        + N_basal_x[None, :]
    )

    alpha_tx = K_tx/N_tx
    norm_p_tx = (
        alpha_tx / (OMEGA + alpha_tx.sum(axis=1)[:, None])
    )

    # Now we have to reinsert the total protein content compatibly with
    # the differential simulation. To do that, we multiply norm_p_tx such that
    # the final size is back to baseline
    factor = y0['ps']/ norm_p_tx[-1] 
    p_tx = norm_p_tx * factor
    
    
    # BUILDING THE RETURN OBJECT
    obs_y0 = {}
    for k, v in y0.items():
        if "d" in k:
            obs_y0[k] = v[ospecs.obs_dendrite_indexes]
        else:
            obs_y0[k] = v[ospecs.obs_spine_arange]

    obs_yt = {k: None for k in y0.keys()}

    obs_yt['ks'] = K_tx[:, ospecs.obs_spine_arange]
    obs_yt['ns'] = N_tx[:, ospecs.obs_spine_arange]
    obs_yt['ps'] = p_tx[:, ospecs.obs_spine_arange]
    obs_yt['ud'] = -jnp.ones_like(obs_yt['ps'])
    obs_yt['us'] = -jnp.ones_like(obs_yt['ps'])

    # breakpoint()
    return (ospecs, obs_y0, obs_yt)


