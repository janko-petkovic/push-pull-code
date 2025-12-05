'''
This script loads a result obtained with fit-model.py and samples the posterior 
distributions of the general parameters using 10 chains, and 100000 samples in total.
The script is called from the pypesto-fit folder via

$ python sample-posteriors.py n_samples

and the sampling result will be saved in

$ output/sampling/sample_APTAM_[n_samples].hdf5
'''

import sys
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

from pypesto import sample, visualize
from pypesto.store import save_to_hdf5

from rdn.fitting.pipeline import *
from rdn.fitting.models import LocalGaussModelTilde, MultiInterface
from rdn.fitting.losses import *


if __name__ == '__main__':


    n_samples = 100001

    # Features of the result to be loaded
    model = LocalGaussModelTilde()
    multi_interface = MultiInterface(model)
    loss_fn = NLLAdast()
    n_starts = 1200
    nsss = [1,3,5,7]

    problem = setup_multi_pypesto_problem(
        nsss,
        multi_interface,
        loss_fn,
        plot_data=False
    )

    # Load the result from the optimization
    result = optimize_multi_problem(
        nsss,
        multi_interface,
        loss_fn,
        problem,
        n_starts,
        mode='multi_fitting',
        force_optimization=False,
    )


    try:
        print('Sampling file already present. Loading old result.')
        result_sampling = read_from_hdf5.read_result(
            filename=f'output/sampling/sample_APTAM_{n_samples}.hdf5',
            problem=True,
            sample=True
        )

    except:
        print(f'No save file with {n_samples} samples. Generating sampling.')

        fix_start_idx = 6

        par_names = problem.x_names
        par_idxes = np.arange(len(par_names))
        par_vals = result.optimize_result.as_list()[0].x

        # problem.unfix_parameters(par_idxes)
        problem.fix_parameters(
            parameter_indices = par_idxes[fix_start_idx:], 
            parameter_vals = par_vals[fix_start_idx:],
        )


        x0s = result.optimize_result.as_list()[0].x
        sampler = sample.AdaptiveParallelTemperingSampler(
            internal_sampler=sample.AdaptiveMetropolisSampler(), n_chains=10
        )
        result_sampling = sample.sample(problem=problem, 
                                        n_samples=n_samples, 
                                        sampler=sampler, 
                                        x0=x0s[:fix_start_idx],
                                        filename=None)

        sample.geweke_test(result_sampling)
        save_to_hdf5.write_result(result=result_sampling,
                                filename=f'output/sampling/sample_APTAM_{n_samples}.hdf5',
                                overwrite=True, problem=True, sample=True)

    print('[SUCCESS] Sampling complete.')