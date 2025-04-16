import math
import numpy as np
from scipy.special import erf
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
import time

from params import params
from datetime import datetime

from twrnn_class import Twoway_coding
from utils import *

# Test
def test_model(model, parameter, N_test): 
    # Generate test data
    bit1_test     = torch.randint(0, 2, (N_test, parameter.N_bits, 1)) 
    bit2_test     = torch.randint(0, 2, (N_test, parameter.N_bits, 1)) 
    # bit1_test     = torch.randint(0, 2, (N_test, parameter.tot_N_bits, 1)).vie
    # bit2_test     = torch.randint(0, 2, (N_test, parameter.tot_N_bits, 1)) 
    noise1_test  = model.sigma1*torch.randn((N_test, parameter.N_channel_use,1))
    noise2_test   = model.sigma2*torch.randn((N_test, parameter.N_channel_use,1))
    
    model.eval() # model.training() becomes False
    N_iter = (N_test//parameter.batch_size) # N_test should be multiple of batch_size
    ber1=0
    bler1=0
    ber2=0
    bler2=0
    power1_acc = np.zeros((parameter.batch_size, parameter.N_channel_use ,1))
    power2_acc = np.zeros((parameter.batch_size, parameter.N_channel_use ,1))
    with torch.no_grad():
        for i in range(N_iter):
            bit1 = bit1_test[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_bits,1) # batch, M,1
            bit2 = bit2_test[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_bits,1)
            noise1 = noise1_test[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_channel_use,1) # batch, T,1
            noise2 = noise2_test[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_channel_use,1) # batch, T,1

            device = parameter.device
            bit1 = bit1.to(device)
            bit2 = bit2.to(device)
            noise1 = noise1.to(device)
            noise2 = noise2.to(device)

            # Forward pass
            X2_hat, X1_hat = model(bit1, bit2, noise1, noise2)

            if parameter.output_type == 'bit_vector':
                ber1_tmp, bler1_tmp = error_rate_bitvector(X1_hat.cpu(), bit1.cpu())
                ber2_tmp, bler2_tmp = error_rate_bitvector(X2_hat.cpu(), bit2.cpu())
                # ber1_tmp, bler1_tmp = error_rate_bitvector(X1_hat.view(-1,parameter.tot_N_bits,1).cpu(), bit1.view(-1,parameter.tot_N_bits,1).cpu())
                # ber2_tmp, bler2_tmp = error_rate_bitvector(X2_hat.view(-1,parameter.tot_N_bits,1).cpu(), bit2.view(-1,parameter.tot_N_bits,1).cpu())
            elif parameter.output_type == 'one_hot_vector':
                ber1_tmp, bler1_tmp = error_rate_onehot(X1_hat.cpu(), bit1.cpu(), parameter.tot_N_bits)
                ber2_tmp, bler2_tmp = error_rate_onehot(X2_hat.cpu(), bit2.cpu(), parameter.tot_N_bits)
                # ber1_tmp, bler1_tmp = error_rate_onehot(X1_hat.view(-1,parameter.tot_N_bits,1).cpu(), bit1.view(-1,parameter.tot_N_bits,1).cpu())
                # ber2_tmp, bler2_tmp = error_rate_onehot(X2_hat.view(-1,parameter.tot_N_bits,1).cpu(), bit2.view(-1,parameter.tot_N_bits,1).cpu())
                
            ber1 = ber1 + ber1_tmp
            ber2 = ber2 + ber2_tmp
            bler1 = bler1 + bler1_tmp
            bler2 = bler2 + bler2_tmp
            
            # Power
            signal1 = model.x1.cpu().detach().numpy()
            power1_acc += signal1**2 
            signal2 = model.x2.cpu().detach().numpy()
            power2_acc += signal2**2 
            
        ber1  = ber1/N_iter
        ber2  = ber2/N_iter
        bler1 = bler1/N_iter
        bler2 = bler2/N_iter
        power1_avg = power1_acc/N_iter
        power2_avg = power2_acc/N_iter

    return ber1, ber2, bler1, bler2, power1_avg, power2_avg