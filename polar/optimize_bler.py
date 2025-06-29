from polarcodes import PolarCode
import numpy as np
import os

# You'll need to install this project: https://github.com/mcba1n/polar-codes/tree/master
# It seemed to work just fine for this basic purpose.
if __name__=='__main__':
    K = 4
    M = 10
    N = 32 # not really necessary
    save_file = 'sim'
    save_to = os.path.join(f'k{K}m{M}n{N}/{save_file}')
    os.makedirs(save_to, exist_ok=True) 
    pc = PolarCode(M,K)
    pc.simulate(save_to=save_to,
                Eb_No_vec=np.array([-1, 1, 5, 10]),
                design_SNR=10,
                manual_const_flag=False,
                min_iterations=10000)