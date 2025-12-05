from enum import auto
import numpy as np
np.random.seed(1)
#Seed 0 -> lognorm explodes
# Seed 2 -> everything works 
# Seed 4 -> similar to data

from ..dendrites import DiffusiveDendrite
from ..spinemodels import LinearSpineModel
from ..visualization import *

def linear_experiment(
        n_spines: int = 1000,
        initial_dendritic_count: int = 1000,
        diffusion_coefficient: float  = 0.01,
        a_beta_in: float = 7,
        b_beta_in: float = 7,
        a_beta_out: float = 7,
        b_beta_out: float = 7,
        time_steps: int = 100
    ) -> tuple:
    '''Simulates the diffusion and exchange dynamics in a dendrite with spines
    with no flux boundary conditions.
    Returns the amounts per spine and dendritic cell at each timepoint.
    '''

    # Free protein distribution (p_out)
    # Include some profile due to influx
    # exponents = np.linspace(0,n_spines, n_spines)
    # p_outs = np.exp(-exponents/n_spines)*100
    # p_outs = np.floor(p_outs).astype(int)

    # No influx, flat steady state
    p_outs = np.ones(n_spines, dtype=int)*initial_dendritic_count

    # Diffusion coefficients. We assume them to be the same
    # for left and right.
    diff_p = np.ones(n_spines) * diffusion_coefficient

    # k_in, k_out distributions
    k_ins = np.random.beta(a_beta_in,b_beta_in,n_spines)
    # k_ins[:] = 0.8
    k_outs = np.random.beta(a_beta_out,b_beta_out,n_spines)

    # Flux boundary conditions
    lf_bc = 0
    rf_bc = 0

    # Time steps for the simulation
    time_steps = 100
    
    spine_model = LinearSpineModel(k_ins = k_ins,
                                   k_outs = k_outs)

    dendrite = DiffusiveDendrite(spine_model = spine_model,
                                 p_outs = p_outs,
                                 diff_p = diff_p,
                                 lf_bc = lf_bc,
                                 rf_bc = rf_bc)

    p_outs_t, p_ins_t = dendrite.simulate(time_steps)

    return p_outs_t, p_ins_t

    # paper_figure(model = spine_model,
    #              p_outs_t = p_outs_t,
    #              p_ins_t = p_ins_t)

    plot_simulation_results(model = spine_model,
                           p_outs_t = p_outs_t,
                           p_ins_t = p_ins_t)

    fit_distr_panel(p_ins_t)
    # autocorrelation_panel(p_ins_t)
    # max_panel(p_ins_t)
                    
