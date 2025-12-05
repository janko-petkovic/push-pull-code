import os

import torch
import numpy as np

from functools import reduce

from fides import BFGS

from pypesto import optimize
from pypesto.engine import MultiProcessEngine
from pypesto.profile import parameter_profile
from pypesto.store import save_to_hdf5, read_from_hdf5


def red_chi_squared(nss, model, result):
    # Load t, x
    t = torch.from_numpy(np.loadtxt(f'data/{nss}Spine_t.txt')).to(torch.float64)
    x = torch.from_numpy(np.loadtxt(f'data/{nss}Spine_x.txt')).to(torch.float64)

    # Create the meshes
    t_mesh, x_mesh = torch.meshgrid(t,x,indexing="ij")

    # Load the target and errors
    data_name = f'{nss}Spine_data'
    data = torch.from_numpy(np.loadtxt(f'data/{data_name}.txt')).to(torch.float64)
    data_err = torch.from_numpy(np.loadtxt(f'data/{data_name}_errs.txt')).to(torch.float64)

    # Retreive p_optim and the nss_weights (needed in the model)
    p_optim = torch.tensor(result.optimize_result.as_list()[0].x)
    nss_weights = model.generate_nss_weights(nss, t, x)

    # Compute the chi squared
    prediction = model(nss_weights, t_mesh, x_mesh, p_optim)
    
    ndof = len(data.flatten()) - len(p_optim) # Sus
    chi2 = (((data - prediction.numpy())/data_err)**2).sum()
    mse = ((data - prediction)**2).mean()
    
    return mse, chi2/ndof



def multi_red_chi_squared(nsss, multi_interface, result):

    if not multi_interface.is_multi: raise ('Multi interface not found!')

    # Import all the data
    tx_pairs = []
    mesh_pairs = []
    datas = []
    data_errs = []

    for nss in nsss:
        t = torch.from_numpy(np.loadtxt(f'data/{nss}Spine_t.txt')).to(torch.float64)
        x = torch.from_numpy(np.loadtxt(f'data/{nss}Spine_x.txt')).to(torch.float64)

        t_mesh, x_mesh = torch.meshgrid(t,x,indexing="ij")

        data_name = f'{nss}Spine_data'
        data = torch.from_numpy(np.loadtxt(f'data/{data_name}.txt')).to(torch.float64)
        data_err = torch.from_numpy(np.loadtxt(f'data/{data_name}_errs.txt')).to(torch.float64)

        tx_pairs.append([t,x])
        mesh_pairs.append([t_mesh, x_mesh])
        datas.append(data)
        data_errs.append(data_err)

    # This concatenate is to be able to use the losses
    datas = torch.concatenate(datas, axis=1)
    data_errs = torch.concatenate(data_errs, axis=1)

    # Retreive p_optim and the nss_weights (needed in the model)
    nsss_weights = multi_interface.generate_nsss_weights(nsss, tx_pairs)
    p_optim = torch.tensor(result.optimize_result.as_list()[0].x)


    # Compute the chi squared
    prediction = multi_interface(True, nsss_weights, mesh_pairs, p_optim)
    

    ndof = len(datas.flatten()) - len(p_optim) # Sus
    chi2 = (((datas - prediction.numpy())/data_errs)**2).sum()
    mse = ((datas - prediction)**2).mean()
    
    return mse, chi2/ndof



def multi_profile_result(nsss, multi_interface, loss_fn, problem, n_starts, result, force_profiling=False):

    if not multi_interface.is_multi: raise ('Multi interface not found!')

    # Save what you found for the love of god
    nsss_str = reduce(lambda str, x: str + f'{x}_', nsss, '')
    data_name = nsss_str + 'Spine_data'
    dirname = f'output/profiled/{multi_interface}/{loss_fn}'
    filename = f'{dirname}/{data_name}_fides_{n_starts}.hdf5'

    # Load if possible
    try:
        if force_profiling: raise RuntimeError('Forcing new profiling')


        profile_idxes = multi_interface.generate_profile_idxes(result)

        print(profile_idxes)
        
        profiled_result = read_from_hdf5.read_result(filename=filename,
                                    problem=True, optimize=True)

    # Run profiling if not possible 
    except Exception as exc:
        print(exc)
        
        engine = MultiProcessEngine(method='spawn')
        optimizer = optimize.FidesOptimizer(hessian_update=BFGS(), verbose=False)

        profile_idxes = multi_interface.generate_profile_idxes(result)

        print(profile_idxes)

        profiled_result = parameter_profile(
            problem, 
            result, 
            optimizer,
            profile_index = profile_idxes,
            next_guess_method='adaptive_step_order_1',
            # profile_options=dict(default_step_size=1),
            engine=engine
        )

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        # For the love of god save what you did
        save_to_hdf5.write_result(result=profiled_result,
                                  filename=filename,
                                  overwrite=True, problem=True, optimize=True)


    return profiled_result