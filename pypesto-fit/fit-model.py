'''
This script fits the selected model on the chosen data.

The parameters are inserted directly in the code, so no arguments
are necessary when calling the script. You just have to run

$ python fit-model.py

calling from inside the pypesto-fit folder.
The result of the optimization will be saved in 

$ output/multi-fitting/[model-name]/[loss-name]/[dataset]_fides_[n_multistarts].hdf5
'''


# Do not let np.linalg choose how many threads to use - this is a hotfix
import os
os.environ['OMP_NUM_THREADS'] = '1'

# Proceed with usual importing
import torch 
torch.manual_seed(0)

import numpy as np
np.random.seed(0)

from rdn.fitting.pipeline import *
from rdn.fitting.models import LocalGaussModelTilde, MultiInterface
from rdn.fitting.losses import *
from rdn.fitting.visualize import *

import matplotlib.pyplot as plt
# plt.style.use('rdn.plotstyles.rdnstyle')


if __name__ == '__main__':

    ### INPUT PARAMETERS ###

    # Select the model
    model = LocalGaussModelTilde()
    multi_interface = MultiInterface(model)

    # Select the loss (Student negative log-likelihood with adaptive degrees of freedom)
    loss_fn = NLLAdast()

    # Number of multi-starts
    n_starts = 1200
    
    # If true: do not load an already present fitted model, when present
    force_optimization = True 

    # Which protocols (identified by the number of stimulations) to be used for the 
    # fitting
    nsss = [1,3,5,7]

    
    ### FITTING ###
    problem = setup_multi_pypesto_problem(nsss, multi_interface, loss_fn,
                                          plot_data=False)

    result = optimize_multi_problem(
        nsss,
        multi_interface,
        loss_fn,
        problem,
        n_starts,
        mode='multi_fitting',
        force_optimization=force_optimization,
    )

    print('[SUCCESS] Fitting complete.')
