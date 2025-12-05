'''
Dendrite implementation with protein undergoing only diffusion.
'''

import numpy as np
from typing import Type, Optional

from .basedendrite import BaseDendrite
from ..spinemodels import BaseSpineModel
from ..random import multi_multinomial

class DiffusiveDendrite(BaseDendrite):
    def __init__(self,
                 spine_model: Type[BaseSpineModel],
                 p_outs: np.ndarray,
                 diff_p: np.ndarray,
                 lf_bc: int = 0,
                 rf_bc: int = 0) -> None:
        '''
        Parameters
        ----------
        spine_model : spine resource exchange model
            defines the equations driving the protein exchange between the
            dendrite and each of the spines

        p_outs : array of int
            starting mean field concentration of proteins in the dendrite

        diff_p : array of float
            values of the protein diffusion coefficient for each point of the
            dendrite

        lf_bc : int
            Left flux boundary condition. In this implementation, it means the
            number of proteins that will be added at the left boundary each
            time step

        rf_bc : int
            As above but at the right boundary.
        '''

        super(DiffusiveDendrite, self).__init__(spine_model,
                                                p_outs)

        self.diff_p = diff_p
        self.lf_bc = lf_bc
        self.rf_bc = rf_bc


    def step(self):
        # This is the point where we can implement an explicit dynamic
        # behaviour of the spine related constants.
        # A complete model, however, should derive this rule from 
        # additional dendritic/spine compartments and not from abstract
        # coefficients
        # spine_model.step()

        # First create the dendritic transitions pvals with the trailing 0s
        # that will be normalized by the multi_multinomial algorythm to account
        # for the probability of a particle staying where it is
        k_ins_t = self.spine_model.compute_k_ins_t(self.p_ins)
        zeros = np.zeros(len(k_ins_t))
        pvalss = np.stack([self.diff_p, self.diff_p, k_ins_t, zeros], axis=0).T

        # Generate new populations
        left, right, entering, staying_out = multi_multinomial(self.p_outs, 
                                                               pvalss).T

        exiting = self.spine_model.generate_exiting(self.p_ins)
        staying_in = self.p_ins - exiting

        # Diffusion
        # Update the distributions with the moved quantities
        staying_out[:-1] += left[1:]
        staying_out[1:] += right[:-1]

        # Apply the flux boundary conditions
        # if the boundaries are empty don't take away anything
        staying_out[0] += self.lf_bc
        staying_out[-1] += self.rf_bc

        ## We cannot have effluxed more than we had
        staying_out[0] = int(max(0, staying_out[0]))
        staying_out[-1] = int(max(0, staying_out[-1]))

        # Boundary correction (account for p_outs that should not leave
        # at the beginning and at the end of the dendrite)
        staying_out[0] += left[0]
        staying_out[-1] += right[-1]

        # Exchange and finalization
        # Generate the proteins exiting the spines
        self.p_outs = staying_out + exiting
        self.p_ins = staying_in + entering

        # Correct for the maximum spine size if present
        if hasattr(self.spine_model, 'etas'):
            overflow = np.maximum(self.p_ins - self.spine_model.etas, 0)
            self.p_ins = self.p_ins - overflow
            self.p_outs = self.p_outs + overflow 
            


    def simulate(self,
                 time_steps: int,
                 p_ins_0: Optional[int] = None) -> list[np.ndarray]:
        '''
        Simulation routine

        Parameters
        ----------
        time_steps: int
        
        p_ins_0: optional int
            Initial value for p_ins. If omitted, the default acceptable steady
            state value will be used.
        '''
        if p_ins_0 is not None:
            self.p_ins = p_ins_0

        p_outs_t = []
        p_ins_t = []

        for _ in range(time_steps):
            self.step()
            p_outs_t.append(self.p_outs)
            p_ins_t.append(self.p_ins)

        p_outs_t = np.stack(p_outs_t, axis=0)
        p_ins_t = np.stack(p_ins_t, axis=0)

        return p_outs_t, p_ins_t

