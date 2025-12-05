'''
The core class of the validation, actually simulating the 
spine evolution
'''

from math import sqrt

import torch
torch.manual_seed(2022)

import numpy as np
np.random.seed(2023)

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from ..fitting.models import BaseModel



class Simulation():

    def __init__(self, 
                 model : BaseModel,
                 model_p_dict : dict,
                 simulation_time : int,
                 spine_number : int,
                 inter_spine_distance : float,
                 stim_indexes : torch.tensor,
                 p_fail_to_uncage: float = 0.) -> None:
        '''
        Parameters
        ----------

        model : instance of BaseModel
            One of the models used to fit the data
        model_p_dict : dict
            Dictionary containing the fitted optimal values for the model 
        simulation_time : int
            Length of the simulation (minutes)
        spine_number : int
            Number of spines in the dendrite
        inter_spine_distance : float
            Distance between adjacent spines (micrometers)
        stim_indexes:
            Indexes for stimulated spines
        '''

        self.model = model
        self.p_dict = model_p_dict

        self.stim_indexes = stim_indexes

        self.t = torch.linspace(0, 
                                int(simulation_time), 
                                int(simulation_time)+1, 
                                dtype=torch.float64)

        self.x = torch.linspace(0, 
                                (spine_number-1)*inter_spine_distance,
                                spine_number,
                                dtype=torch.float64)

        self.t_mesh, self.x_mesh = torch.meshgrid((self.t,self.x), indexing='ij')

        self.x_stim = self.x.clone()[stim_indexes]

        self.p_fail_to_uncage = p_fail_to_uncage

        self.p_dict['Omega'] = (spine_number*inter_spine_distance) / self.p_dict['Chi']

    
    def run(self, n_samples : int, seed: int = 2025) -> tuple:
        '''
        Parameters
        ----------

        n_samples : int
            Number of samplings for the initial conditions

        Returns
        -------

        basal_sizes_batch, sizes_tx_batch, rel_sizes_tx_batch
            Three batches of dimension (t+1, x, n_samples) with the resulting
            basal, absolute and relative spine sizes. Yes, the basals are the
            same for all the time points but for now we accept this 
            memory inefficiency and trade it with usability.

            Note: we always drop minutes 0 and 1 (t-1) because the simulation is
            not reliable there (40 minutes + 0th minute - 0th and 1st minute)
        '''

        basal_sizes_batch = []
        sizes_tx_batch = []
        rel_sizes_tx_batch = []

        np.random.seed(seed)
        
        for _ in range(n_samples):
            
            # Basal Kbs and Nbs. To understand the parameters look at how
            # scipy.stats and numpy parametrize the lognormal distribution
            
            log_Kbs, log_Nbs = np.random.multivariate_normal(
                                    self.p_dict['mu_log_K_N'],
                                    self.p_dict['cov_log_K_N'],
                                    len(self.x)).T

            Kbs = np.exp(log_Kbs)
            Nbs = np.exp(log_Nbs)

            self.p_dict['Kbs'] = torch.from_numpy(Kbs).to(torch.float64)
            self.p_dict['Nbs'] = torch.from_numpy(Nbs).to(torch.float64)

            x_stim = self.x_stim.clone()

            # Failure to uncage
            if self.p_fail_to_uncage:
                m = torch.rand(len(self.x_stim))
                x_stim = x_stim[torch.where(m>self.p_fail_to_uncage)]
                

            # This has to be random for each run
            sizes_tx, basal_sizes = self.model.validation_forward(self.t_mesh,
                                                                  self.x_mesh,
                                                                  x_stim,
                                                                  self.p_dict)

            basal_sizes_batch.append(basal_sizes)
            sizes_tx_batch.append(sizes_tx)
            rel_sizes_tx_batch.append(sizes_tx/basal_sizes)

        basal_sizes_batch = torch.stack(basal_sizes_batch, dim=2)
        sizes_tx_batch = torch.stack(sizes_tx_batch, dim=2)
        rel_sizes_tx_batch = torch.stack(rel_sizes_tx_batch, dim=2)

        # We overwrite the first two time points with nans, because the model is
        # not valid in that regime. Also, we want the coder to still use indexes
        # as minutes, so we avoid returning only the valid portion of the
        # result.
        basal_sizes_batch[:2] = float('nan')
        sizes_tx_batch[:2] = float('nan')
        rel_sizes_tx_batch[:2] = float('nan')

        return  (basal_sizes_batch, 
                 sizes_tx_batch,
                 rel_sizes_tx_batch)


    def visualize_run(self, n_samples : int, t_idx : int) -> None:
        '''
        A wrapper of run that interactively visualizes the simulation result.
        Mainly used for debugging.
        '''
        _, _, rel_sizes_tx_batch = self.run(n_samples)

        rel_sizes_mean = rel_sizes_tx_batch.mean(axis=2)
        rel_sizes_std = rel_sizes_tx_batch.std(axis=2)/sqrt(len(rel_sizes_tx_batch))*3

        fig, ax = plt.subplots()

        # line, = ax.plot(x, rel_sizes_mean[0])
        ax.errorbar(self.x, rel_sizes_mean[t_idx], rel_sizes_std[t_idx], fmt='.')
        ax.errorbar(self.x_stim, 
                    rel_sizes_mean[t_idx][self.stim_indexes], 
                    rel_sizes_std[t_idx][self.stim_indexes],
                    fmt='.')
        ax.axhline(y=1, linestyle=':', c='tab:gray')
        ax.set_ylim(0,3)

        fig.subplots_adjust(bottom=0.25)
        ax_slider = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        t_slider = Slider(
            ax = ax_slider,
            label='time',
            valstep=self.t,
            valmin=2,
            valmax=self.t.max(),
            valinit=0
        )

        def update(val):
            ax.clear()
            ax.errorbar(self.x, rel_sizes_mean[int(val)], rel_sizes_std[int(val)], fmt='.')
            ax.errorbar(self.x_stim, 
                        rel_sizes_mean[int(val)][self.stim_indexes], 
                        rel_sizes_std[int(val)][self.stim_indexes],
                        fmt='.')
            ax.set_ylim(0,3)
            ax.axhline(y=1, linestyle=':', c='tab:gray')

            fig.canvas.draw_idle()


        t_slider.on_changed(update)

        plt.show()


    def run_return_all(self, n_samples : int, seed: int = 2025) -> tuple:
        '''
        Utility associated with the model validation_run_return_all. Basically
        a wrapper for ease of use, just like Simulation.run.

        Parameters
        ----------

        n_samples : int
            Number of samplings for the initial conditions

        Returns
        -------

        tuple consisting of: 
            basal_sizes_batch, 
            sizes_tx_batch,
            rel_sizes_tx_batch,
            ks_tx_batch,
            basal_ks_batch,
            ns_tx_batch,
            basal_ns_batch
        '''

        basal_sizes_batch = []
        sizes_tx_batch = []
        rel_sizes_tx_batch = []
        ks_tx_batch = []
        basal_ks_batch = []
        ns_tx_batch = []
        basal_ns_batch = []

        np.random.seed(seed)

        for _ in range(n_samples):
            
            # Basal Kbs and Nbs. To understand the parameters look at how
            # scipy.stats and numpy parametrize the lognormal distribution
            
            log_Kbs, log_Nbs = np.random.multivariate_normal(
                                    self.p_dict['mu_log_K_N'],
                                    self.p_dict['cov_log_K_N'],
                                    len(self.x)).T

            Kbs = np.exp(log_Kbs)
            Nbs = np.exp(log_Nbs)

            self.p_dict['Kbs'] = torch.from_numpy(Kbs).to(torch.float64)
            self.p_dict['Nbs'] = torch.from_numpy(Nbs).to(torch.float64)

            x_stim = self.x_stim.clone()

            # Failure to uncage
            if self.p_fail_to_uncage:
                m = torch.rand(len(self.x_stim))
                x_stim = x_stim[torch.where(m>self.p_fail_to_uncage)]
                

            # This has to be random for each run
            run_result = self.model.validation_forward_return_all(
                self.t_mesh,
                self.x_mesh,
                x_stim,
                self.p_dict
                )

            sizes_tx, basal_sizes, K_tx, Kb_tx, N_tx, Nb_tx = run_result

            basal_sizes_batch.append(basal_sizes)
            sizes_tx_batch.append(sizes_tx)
            rel_sizes_tx_batch.append(sizes_tx/basal_sizes)

            ks_tx_batch.append(K_tx)
            basal_ks_batch.append(Kb_tx)
            ns_tx_batch.append(N_tx)
            basal_ns_batch.append(Nb_tx)

        basal_sizes_batch = torch.stack(basal_sizes_batch, dim=2)
        sizes_tx_batch = torch.stack(sizes_tx_batch, dim=2)
        rel_sizes_tx_batch = torch.stack(rel_sizes_tx_batch, dim=2)

        ks_tx_batch = torch.stack(ks_tx_batch, dim=2)
        basal_ks_batch = torch.stack(basal_ks_batch, dim=2)
        ns_tx_batch = torch.stack(ns_tx_batch, dim=2)
        basal_ns_batch = torch.stack(basal_ns_batch, dim=2)


        # We overwrite the first two time points with nans, because the model is
        # not valid in that regime. Also, we want the coder to still use indexes
        # as minutes, so we avoid returning only the valid portion of the
        # result.
        basal_sizes_batch[:2] = float('nan')
        sizes_tx_batch[:2] = float('nan')
        rel_sizes_tx_batch[:2] = float('nan')

        ks_tx_batch[:2] = float('nan')
        basal_ks_batch[:2] = float('nan')
        ns_tx_batch[:2] = float('nan')
        basal_ns_batch[:2] = float('nan')
        
        return  (basal_sizes_batch, 
                 sizes_tx_batch,
                 rel_sizes_tx_batch,
                 ks_tx_batch,
                 basal_ks_batch,
                 ns_tx_batch,
                 basal_ns_batch)


    def run_r_delta_r(self, n_samples : int, seed: int = 2025) -> tuple:
        '''
        Parameters
        ----------

        n_samples : int
            Number of samplings for the initial conditions

        Returns
        -------

        basal_sizes_batch, sizes_tx_batch, rel_sizes_tx_batch
            Three batches of dimension (t+1, x, n_samples) with the resulting
            basal, absolute and relative spine sizes. Yes, the basals are the
            same for all the time points but for now we accept this 
            memory inefficiency and trade it with usability.

            Note: we always drop minutes 0 and 1 (t-1) because the simulation is
            not reliable there (40 minutes + 0th minute - 0th and 1st minute)
        '''

        basal_rs_batch = []
        delta_rs_batch = []

        np.random.seed(seed)

        for _ in range(n_samples):
            
            # Basal Kbs and Nbs. To understand the parameters look at how
            # scipy.stats and numpy parametrize the lognormal distribution
            
            log_Kbs, log_Nbs = np.random.multivariate_normal(
                                    self.p_dict['mu_log_K_N'],
                                    self.p_dict['cov_log_K_N'],
                                    len(self.x)).T

            Kbs = np.exp(log_Kbs)
            Nbs = np.exp(log_Nbs)

            self.p_dict['Kbs'] = torch.from_numpy(Kbs).to(torch.float64)
            self.p_dict['Nbs'] = torch.from_numpy(Nbs).to(torch.float64)

            x_stim = self.x_stim.clone().type(torch.int)

            # Failure to uncage
            if self.p_fail_to_uncage:
                m = torch.rand(len(self.x_stim))
                x_stim = x_stim[torch.where(m>self.p_fail_to_uncage)]
                

            # This has to be random for each run
            basal_rs, delta_rs_t = self.model.validation_forward_r_delta_r(
                self.t_mesh,
                self.x_mesh,
                x_stim,
                self.p_dict
            )

            basal_rs_batch.append(basal_rs)
            delta_rs_batch.append(delta_rs_t)

        basal_rs_batch = torch.stack(basal_rs_batch, dim=1)
        delta_rs_batch = torch.stack(delta_rs_batch, dim=1)

        # We overwrite the first two time points with nans, because the model is
        # not valid in that regime. Also, we want the coder to still use indexes
        # as minutes, so we avoid returning only the valid portion of the
        # result.
        basal_rs_batch[:2] = float('nan')
        delta_rs_batch[:2] = float('nan')

        return  (basal_rs_batch, delta_rs_batch)