'''
Auxiliary methods for the pypesto fit
'''
import os
import torch
import numpy as np
from functools import partial, reduce

import pypesto
import pypesto.optimize as optimize
from fides import BFGS
from pypesto.store import save_to_hdf5, read_from_hdf5

from .visualize import plot_data_2d




def objective_fun(loss_of_p, p):
        
    # convert parameter vector to torch
    p = torch.tensor(p, requires_grad=True)

    # compute the loss
    return loss_of_p(p).item()



def objective_grad(loss_of_p, p):
    
    # convert parameter vector to torch
    p = torch.tensor(p, requires_grad=True)

    # compute the loss
    loss = loss_of_p(p)

    # compute the gratients relative to p
    loss.backward()

    return p.grad.detach().numpy()



def setup_multi_pypesto_problem(nsss, multi_interface, loss_fn, plot_data=False):
    """
    Creates a pyPESTO problem for the multi_interface
    """

    if not multi_interface.is_multi: raise ('Multi interface not found!')

    # Import all the data
    tx_pairs = []
    mesh_pairs = []
    ndofs = []
    datas = []
    data_errs = []

    for nss in nsss:
        t = torch.from_numpy(np.loadtxt(f'binned-data/{nss}Spine_t.txt')).to(torch.float64)
        x = torch.from_numpy(np.loadtxt(f'binned-data/{nss}Spine_x.txt')).to(torch.float64)
        ndof = torch.from_numpy(np.loadtxt(f'binned-data/{nss}Spine_counts.txt')).to(torch.float64)


        t_mesh, x_mesh = torch.meshgrid(t,x,indexing="ij")
        ndof_mesh = ndof.tile((len(t), 1))
        
        data_name = f'{nss}Spine_data'
        data = torch.from_numpy(np.loadtxt(f'binned-data/{data_name}.txt')).to(torch.float64)
        data_err = torch.from_numpy(np.loadtxt(f'binned-data/{data_name}_errs.txt')).to(torch.float64)
        

        # Quick data visualization
        if plot_data: plot_data_2d(data)

        tx_pairs.append([t,x])
        ndofs.append(ndof_mesh)
        mesh_pairs.append([t_mesh, x_mesh])
        datas.append(data)
        data_errs.append(data_err)

    # This concatenate is to be able to use the losses
    datas = torch.concatenate(datas, axis=1)
    data_errs = torch.concatenate(data_errs, axis=1)
    ndofs  = torch.concatenate(ndofs, axis=1)

    p_names = multi_interface.generate_parameter_names(nsss, tx_pairs)
    p_scales = ['log10']*len(p_names)
    lbs, ubs = multi_interface.generate_boundary_conditions(tx_pairs)

    # Problem instantiation: functions
    # Create the spine mass mask for A
    nsss_weights = multi_interface.generate_nsss_weights(nsss, tx_pairs)

    # Create the stack: forward -> loss -> fun/grad
    F_of_p = partial(multi_interface, True, nsss_weights, mesh_pairs)

    if loss_fn.__str__() == 'NLLAdast': 
        loss_of_p = partial(loss_fn, F_of_p, datas, data_errs, ndofs)
    else: 
        loss_of_p = partial(loss_fn, F_of_p, datas, data_errs)

    part_objective_fun = partial(objective_fun, loss_of_p)
    part_objective_grad = partial(objective_grad, loss_of_p)
    
    objective = pypesto.Objective(fun=part_objective_fun, 
                                  grad=part_objective_grad)

    startpoint_method = pypesto.startpoint.latin_hypercube

    p_test = multi_interface.generate_parameter_guess(nsss, tx_pairs).numpy()
    # objective.check_grad_multi_eps(p_test)
    
    # create pypesto problem object
    problem = pypesto.Problem(objective=objective,  # objective function
                              lb=lbs,  # lower bounds
                              ub=ubs,  # upper bounds
                              x_names=p_names,  # parameter names
                              x_scales=p_scales, # parameter scale
                              startpoint_method=startpoint_method)

    # # quick problem check: uncomment this when debugging
    # p_test = multi_interface.generate_parameter_guess(nsss, tx_pairs).numpy()
    # print(problem.objective.call_unprocessed(x=p_test,sensi_orders=(0,1,), mode='mode_fun'))

    return problem




def optimize_multi_problem(nsss, multi_interface, loss_fn, problem, n_starts,
    mode='multi_fitting', force_optimization=False, old_result=None):
    
    nsss_str = reduce(lambda str, x: str + f'{x}_', nsss, '')
    data_name = nsss_str + 'Spine_data'
    dirname = f'output/{mode}/{multi_interface}/{loss_fn}'
    filename = f'{dirname}/{data_name}_fides_{n_starts}.hdf5'

    try:
        if old_result:
            filename = f'{dirname}/{old_result}'
            result = read_from_hdf5.read_result(filename=filename,
                                    problem=True, optimize=True)
            print(f'Reading given old result: {old_result}')

        else:
            if force_optimization: 
                raise RuntimeError('Forcing new optimization (new default name)')

            result = read_from_hdf5.read_result(filename=filename,
                                                problem=True, optimize=True)
            print(f'Result already present: {filename}')

    except Exception as exc:

        print(exc)
        
        engine = pypesto.engine.MultiProcessEngine(method='spawn')
        # engine = pypesto.engine.SingleCoreEngine()

        optimizer = optimize.FidesOptimizer(hessian_update=BFGS(), verbose=False)
        result = optimize.minimize(problem=problem, 
                                   optimizer=optimizer, 
                                   n_starts=n_starts,
                                   engine=engine,
                                   progress_bar=True,
                                   )

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        save_to_hdf5.write_result(result=result,
                                  filename=filename,
                                  overwrite=True, problem=True, optimize=True)

    return result
