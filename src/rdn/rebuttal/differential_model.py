from functools import partial

import jax
import jax.numpy as jnp
from jax.lax.linalg import tridiagonal_solve
# from jax_tqdm import scan_tqdm

import matplotlib.pyplot as plt

from rdn.rebuttal.parameters import Experiment


def bell_kernel(xs, mu, sigma):
    return jnp.exp(-((xs - mu) ** 2) / sigma**2)


def setup_compartment_geometry(parameters):
    dx = parameters.integration.dx
    cpm = int(1 / dx)  # compartments per micron
    n_spines = parameters.dendrite.n_spines

    n_dendrite_compartments = cpm * (n_spines + 1) + 1
    dendrite_indexes = jnp.arange(n_dendrite_compartments)
    spine_indexes = jnp.array([cpm * (i + 1) for i in range(n_spines)])

    return (n_dendrite_compartments, dendrite_indexes, spine_indexes)


def generate_diffusion_operator(parameters, species_name):
    """Laplacian * D"""
    # Simple one-dimensional diffusion structure
    n_compartments, *_ = setup_compartment_geometry(parameters)
    D = parameters.species[species_name].diffusion_coeff
    dx = parameters.integration.dx

    diffusion_operator = (
        -2 * jnp.eye(n_compartments)
        + jnp.diag(jnp.ones(n_compartments - 1), 1)
        + jnp.diag(jnp.ones(n_compartments - 1), -1)
    )

    # No flux boundary conditions
    diffusion_operator = diffusion_operator.at[0, 0].set(-1)
    diffusion_operator = diffusion_operator.at[-1, -1].set(-1)

    return D * diffusion_operator / dx**2


@jax.jit
def tridiag_matvec(sub, diag, sup, y):
    # sub, sup have length N-1; diag has length N
    out = diag * y
    out = out.at[:-1].add(sup * y[1:])
    out = out.at[1:].add(sub * y[:-1])
    return out


def setup_dendrite_initial_conditions(key, parameters) -> tuple:
    """Start with the assumption one spine every 2 microns.
    Returns
    -------
    tuple: (ud, us, kd, ks, nu, ns, pd, ps, spine_sizes), spine_indexes
    """

    # aliases
    species = parameters.species
    n_spines = parameters.dendrite.n_spines
    u_conc = species.unphosphorylated.initial_concentration

    geom = setup_compartment_geometry(parameters)
    n_dendritic_compartments, dendrite_indexes, spine_indexes = geom

    ud = jnp.ones(n_dendritic_compartments) * u_conc

    # Sample the kinases and phosphatases
    log_mu = jnp.array(
        [species[s].log_mu for s in ("kinases", "phosphatases")]
    )
    log_cov = jnp.array(
        (
            (species.kinases.log_var, species.kinases.log_cross_cov),
            (species.kinases.log_cross_cov, species.phosphatases.log_var),
        )
    )

    log_ksns = jax.random.multivariate_normal(
        key, log_mu, log_cov, shape=(n_spines,)
    ).T
    ks, ns = jnp.exp(log_ksns)

    ############### this is the tricky part
    us = jnp.ones(n_spines) * u_conc

    ps = us * ks / ns
    spine_sizes = ps**parameters.dendrite.alpha

    return (
        {
            "ud": ud,
            "us": us,
            "ks": ks,
            "ns": ns,
            "ps": ps,
            "spine_sizes": spine_sizes,
        },
        dendrite_indexes,
        spine_indexes,
    )


def diffusion_step(y0, lm_p, dm, um_p, ln, dn, un):
    """Notice that the dt has already been included in the operators.
    For now I am keeping the parametrization like this, with _p meaning
    that we have to pad the diagonals as "explained" in the jax reference
    for tridiagonal_solve.

    Remember to use concentrations, and not quantities.
    """

    b = tridiag_matvec(ln, dn, un, y0)
    return tridiagonal_solve(lm_p, dm, um_p, b.reshape(-1, 1))[:, 0]


def reaction_step(y0: jax.Array, rate_matrix_dt):
    """
    We will use this in a vectorized way!
    y0: shape (n_species,) (we take it, for example, from
    u[spine_indexes]

    rate_matrix: shape (n_species, n_species)

    Remember to use concentrations, and not quantities.
    """
    # 1st order
    # return y0 + rate_matrix_dt @ y0

    # 2nd order
    # return y0 + rate_matrix_dt @ y0 + rate_matrix_dt@rate_matrix_dt @ y0/2

    # Exact solution
    return jax.scipy.linalg.expm(rate_matrix_dt) @ y0


def setup_rnd_integrator(parameters, spine_indexes, y0, uncaging_mask):
    """
    y0: initial conditions, given as a dictionary. This guys should not take
    the initial conditions in theory, but I need them to integrate the
    catalysts after they are stimulated I think. Dirty but it will do for now.

    The notation is taken from Claudio's explanation.
    In the diffusion step we are solving with Crank-Nicolson

    (1 - dt/2 L) y(n+1) = (1 + dt/2 L) y(n)
    ------------          ------------
         M                      N

    Each of the matrices will be tridiagonal, with a respective
    l, d, and u diagonals. We can use this to speed up the computations.
    Notice that if we are integrating for dt/2 (strang splitting),
    we will have dt/4 in the end.
    """
    dt = parameters.integration.dt

    # Set up the diffusion-solution operators
    crank_nicolson_steppers = {}
    n_dendritic_compartments, *_ = setup_compartment_geometry(parameters)

    for species_name, species_pars in parameters.species.items():
        if species_pars.diffusion_coeff != 0:
            # Laplacian multiplied by the diffusion coefficient
            DL = generate_diffusion_operator(parameters, species_name)

            # Left and right CN operators
            M = jnp.eye(n_dendritic_compartments) - dt / 4 * DL * 2
            N = jnp.eye(n_dendritic_compartments) + dt / 4 * DL * 0

            # diagonals of the tridiagonal operators M and N
            lm, dm, um = (
                jnp.diagonal(M, -1),
                jnp.diagonal(M, 0),
                jnp.diagonal(M, 1),
            )
            ln, dn, un = (
                jnp.diagonal(N, -1),
                jnp.diagonal(N, 0),
                jnp.diagonal(N, 1),
            )

            # Padded diagonals for jax.lax.linalg.tridiag_solve
            lm_p, um_p = jnp.insert(lm, 0, 0), jnp.append(um, 0)

            stepper_parameter_dict = {
                "lm_p": lm_p,
                "dm": dm,
                "um_p": um_p,
                "ln": ln,
                "dn": dn,
                "un": un,
            }

            crank_nicolson_steppers[species_name] = partial(
                diffusion_step, **stepper_parameter_dict
            )

    # Quality of life
    # Locals
    par_un = parameters.species.unphosphorylated
    par_k = parameters.species.kinases
    par_ph = parameters.species.phosphatases
    unc_locations = jnp.array(
        parameters.experiment.uncaging_protocol.spine_locations
    )
    n_spines = parameters.dendrite.n_spines

    # Vectorized reaction stepper
    v_reaction_step = jax.vmap(reaction_step)

    # Uncaging contriutions machinery
    spine_positions = jnp.arange(n_spines)

    # I am not sure this is the best way to implement this, but
    # the performance should not really tank that much, I hope XLA
    # just makes it an expression evaluation for the None case that
    # gets compiled only once.
    v_bell_kernel = jax.vmap(bell_kernel, (None, 0, None))

    k_stim_contributions = v_bell_kernel(
        spine_positions,
        unc_locations,
        par_k.sigma
    ) * par_k.delta_stim

    n_stim_contributions = v_bell_kernel(
        spine_positions,
        unc_locations,
        par_ph.sigma
    ) * par_ph.delta_stim

    def fori_stepper(idx, y):
        """
        This integrate between two adjacent timesteps in the big loop.
        Notice that we have always to pass it the offset variable (first of the
        tuple idxoff_y) so that the uncaging time array is indexed properly
        throughout subsequent integrations.

        NOTE: the uncatging_array now must by a 2 dim array
        """

        # This is either a None or a jax.Array
        m = uncaging_mask[idx]

        new_y = {}

        y["ks"] += jnp.where(m[:, None], k_stim_contributions,
                            jnp.zeros(n_spines)).sum(axis=0)
        y["ns"] += jnp.where(m[:, None], n_stim_contributions,
                            jnp.zeros(n_spines)).sum(axis=0)


        # First diffusion dt/2
        inter_ud = crank_nicolson_steppers["unphosphorylated"](y["ud"])

        # # Then reaction for dt
        ys_react = jnp.stack(
            (
                y["ps"],
                y["us"],
                inter_ud[spine_indexes],
                (y["ks"] - y0["ks"]),
                (y["ns"] - y0["ns"]),
            )
        ).T

        rate_matrixs_dt = (
            jnp.array(
                [
                    (
                        (-n * par_ph.k_cat, k * par_k.k_cat, 0, 0, 0),
                        (
                            n * par_ph.k_cat,
                            -(k * par_k.k_cat + par_un.k_out),
                            par_un.k_in,
                            0,
                            0,
                        ),
                        (0, par_un.k_out, -par_un.k_in, 0, 0),
                        (0, 0, 0, -1 / par_k.tau, 0),
                        (0, 0, 0, 0, -1 / par_ph.tau),
                    )
                    for n, k in zip(y["ns"], y["ks"])
                ]
            )
            * dt
        )

        res = v_reaction_step(ys_react, rate_matrixs_dt).T
        new_y["ps"], new_y["us"] = res[:2]
        new_y["ks"] = res[3] + y0["ks"]
        new_y["ns"] = res[4] + y0["ns"]
        inter_ud = inter_ud.at[spine_indexes].set(res[2])

        # # Finally diffusion for dt/2
        new_y["ud"] = crank_nicolson_steppers["unphosphorylated"](inter_ud)

        # Default pass
        new_y["spine_sizes"] = y["spine_sizes"]

        return new_y

    return fori_stepper


def setup_rdn_uncaging_mask(key, parameters, y0):

    dt = parameters.integration.dt
    obs_time_factor = Experiment.time_unit_to_factor(
        parameters.experiment.time_unit
    )
    unc_freq = parameters.experiment.uncaging_protocol.frequency
    unc_locations = parameters.experiment.uncaging_protocol.spine_locations


    # THIS NEEDS CALIBRATION for quantitative predictions
    stim_locations = parameters.experiment.uncaging_protocol.spine_locations
    # breakpoint()
    x = y0['ps'][jnp.array(stim_locations)]
    ps_uncaging = 1- jnp.exp(-x**2/1000000000)

    assert dt < 1 / unc_freq, (
        "dt too big in relation to the uncaging frequency"
    )

    n_timesteps = int(
        max(parameters.experiment.observation.time_range)
        * obs_time_factor
        / dt
    )

    if not parameters.experiment.uncaging_protocol.active:
        ### EXIT 1
        # return -jnp.ones(n_timesteps)
        return jnp.zeros((n_timesteps, len(unc_locations))).astype(bool)
    
    unc_duration = parameters.experiment.uncaging_protocol.duration
    unc_duration_in_s = unc_duration * obs_time_factor

    n_uncs = int(unc_freq * unc_duration_in_s)

    if parameters.experiment.uncaging_protocol.with_failure:
        was_uncaged_masks = jax.random.binomial(
            key, 1, ps_uncaging, shape=(n_uncs, len(unc_locations))
        ).astype(bool)
    else:
        was_uncaged_masks = jnp.ones(
            shape=(n_uncs, len(unc_locations))
        ).astype(bool)


    n_steps_per_unc = int(1 / (unc_freq * dt))
    unc_mask = jnp.zeros((n_timesteps, len(unc_locations))).astype(bool)

    # This is where we put the locations
    for i, m in enumerate(was_uncaged_masks):
        unc_mask = unc_mask.at[i*n_steps_per_unc].set(m)

    # breakpoint()
    ### EXIT 2
    return unc_mask
