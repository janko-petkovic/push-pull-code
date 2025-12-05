import matplotlib.pyplot as plt
import numpy as np


from ..spinemodels import LogisticSpineModel
from ..dendrites import DiffusiveDendrite
from ..visualization import plot_simulation_results
from ..visualization import max_panel

def logistic_experiment(n_spines: int = 1000):

    # Free protein distribution (p_out)
    exponents = np.linspace(0,n_spines, n_spines)
    p_outs = np.exp(-exponents/n_spines)*1000
    p_outs = np.floor(p_outs).astype(int)
    #p_outs = np.ones(n_spines, dtype=int)*10000
    
    # Diffusion coefficients. We assume them to be
    # the same for left and right
    diff_p = np.ones(n_spines) * 0.01

    # k_in, k_out distributions
    k_ins = np.random.beta(13,13,n_spines)*0.0
    k_outs = np.random.beta(13, 13,n_spines)*0.001
    #k_ins = np.ones(n_spines)*0.1
    #k_outs = np.ones(n_spines)*0.001

    # eta distribution
    etas = np.random.lognormal(np.log(100),.3,n_spines).astype(int)*10
    #etas = np.ones(n_spines, dtype=int).astype(int)*200

    # Flux boundary conditions
    lf_bc = 0
    rf_bc = 0

    # Simulation time steps
    time_steps = 100


    spine_model = LogisticSpineModel(k_ins = k_ins,
                                     k_outs = k_outs,
                                     etas = etas)

    dendrite = DiffusiveDendrite(spine_model = spine_model,
                                 p_outs = p_outs,
                                 diff_p = diff_p,
                                 lf_bc = lf_bc,
                                 rf_bc = rf_bc)

    p_outs_t, p_ins_t = dendrite.simulate(time_steps,0)
    
    max_panel(p_ins_t)
    # plot_simulation_results(model = spine_model,
    #                         p_outs_t = p_outs_t,
    #                         p_ins_t = p_ins_t)
