'''
Implementation for the linear spine exchange equation, namely

\dot{p_\in} = k_in p_out - k_out p_in
'''

import numpy as np

from .basespinemodel import BaseSpineModel

class LinearSpineModel(BaseSpineModel):
    def __init__(self,
                 k_ins: np.ndarray,
                 k_outs: np.ndarray) -> None:
        
        super(LinearSpineModel, self).__init__(k_ins,
                                               k_outs)

    def compute_pins(self, 
                     p_outs: np.ndarray) -> np.ndarray:
        '''
        Compute the mean field p_ins in equilibrium with the given
        p_outs.

        Parameters
        ----------
        p_outs: numpy array
            Dendritic proteins for each dendritic block

        Returns
        -------
        numpy array
            Spine proteins for each spine
        
        Caveat
        ------
        This is the only method where we are losing mass as we are flooring
        the result.
        '''

        ais = self.k_ins/self.k_outs
        return np.floor(p_outs * ais).astype(int)


    def compute_k_ins_t(self,
                        p_ins: np.ndarray):
        '''This method is trivial for the linear exchange model.'''
       
        return self.k_ins


    def generate_exiting(self,
                         p_ins: np.ndarray) -> np.ndarray:
        '''The number of proteins leaving the dendrite'''

        return np.random.binomial(n = p_ins,
                                  p = self.k_outs)

if __name__ == '__main__':
    sm = LinearSpineModel(0.5,0.5)
    sm.compute_k_ins_t(100)
    sm.generate_exiting(0.4) 
    print('all good!')
