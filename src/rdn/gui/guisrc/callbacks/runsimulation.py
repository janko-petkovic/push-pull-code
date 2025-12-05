import ast
import tkinter as tk
import torch

from rdn.fitting.models import LocalGaussModelTilde
from rdn.validation import Simulation


def run_simulation(input_entries: list):
    '''
    Parameters
    ----------
    input_entries : list
        List of the entries coming from the GUI where we are taking the
        simulations from
    '''
        

    # Parse the input_entries[].get()
    simulation_time = int(input_entries[0].get())
    inter_spine_distance = int(input_entries[1].get())
    spine_number = int(input_entries[2].get())

    
    start_stim_idx = int(input_entries[3].get())
    n_stims = int(input_entries[4].get())
    stride_stims = int(input_entries[5].get())

    stim_idxes = torch.tensor([start_stim_idx + i * stride_stims
                              for i in range(n_stims)])

    # stim_idxes = torch.tensor([50,53,54,60])

    # Build the pardict
    try:
        model_p_dict = {
            'Chi' : float(input_entries[6].get()),
            'Pi' : float(input_entries[7].get()),

            'Ks' : float(input_entries[8].get()),
            'sigma_K' : float(input_entries[9].get()),
            'tau_K' : float(input_entries[10].get()),

            'Ns' : float(input_entries[12].get()),
            'sigma_N' : float(input_entries[13].get()),
            'tau_N' : float(input_entries[14].get()),

            'mu_log_K_N' : [float(input_entries[11].get()),
                            float(input_entries[15].get())],

            'cov_log_K_N' : [[float(input_entries[16].get()),
                            float(input_entries[17].get())],
                            [float(input_entries[16].get()),
                            float(input_entries[17].get())]]
        }

    # model_p_dict_str = input_entries[6].get('1.0',tk.END).replace('\n', ' ')
    # model_p_dict_str = '{' + model_p_dict_str + '}' model_p_dict =
    # ast.literal_eval(model_p_dict_str)
    except:
        print('Using fallback model_p_dict')
        model_p_dict = {
        "Chi" : 1,
        "Ks" : 30000.000000000415,
        "Ns" : 36551.83105199166,
        "tau_K" : 5.143176990574988,
        "sigma_K" : 1.000000000000009,
        "tau_N" : 5.408911563942017,
        "sigma_N" : 1.1980624666339608,
        "Pi" : 59429504.467310235,
        "mu_log_K_N" : [8.10676469, 8.82097831],
        "cov_log_K_N" : [[0.32407014, 0.28441916],
        [0.28441916, 0.35623684]],
        }

    simulation = Simulation(model = LocalGaussModelTilde(),
                            model_p_dict = model_p_dict,
                            simulation_time = simulation_time,
                            spine_number = spine_number,
                            inter_spine_distance = inter_spine_distance,
                            stim_indexes = stim_idxes)
    # simulation.visualize_run(100,2)
    
    return simulation.run(100)
    


