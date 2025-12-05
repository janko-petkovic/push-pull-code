'''
Implementation for the logistic spine exchange equation
'''

import numpy as np

from .basespinemodel import BaseSpineModel

class LogisticSpineModel(BaseSpineModel):
    def __init__(self,
                 k_ins: np.ndarray,
                 k_outs: np.ndarray,
                 etas: np.ndarray) -> None:
        
        super(LogisticSpineModel, self).__init__(k_ins,
                                                 k_outs)
        self.etas = etas
        
    def compute_pins(self, p_outs):
        '''
        Compute the mean field p_ins_mf in equilibrium with the given
        p_outs_mf.

        Notes
        -----
        The equation can be written in a simpler form but this way I avoid
        problems when setting k_ins to 0.
        
        Caveat
        ------
        This is the only method where we are losing mass as we are flooring
        the result.
        '''
        return np.floor(self.k_ins*p_outs / (self.k_outs + self.k_ins*p_outs/self.etas)).astype(int)


    def compute_k_ins_t(self, p_ins):
        '''The total k_in given p_ins'''
        return self.k_ins * (1-p_ins/self.etas)


    def generate_exiting(self,
                         p_ins: np.ndarray) -> np.ndarray:
        '''The number of proteins leaving the dendrite'''

        return np.random.binomial(n = p_ins,
                                  p = self.k_outs)


if __name__ == '__main__':
    sm = LogisticSpineModel(0.5,0.5, 10)
    sm.compute_k_ins_t(100)
    sm.generate_exiting(0.4) 
    print('all good!')
