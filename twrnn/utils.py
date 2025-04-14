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

# Convert the `bit vector' with (batch,K,1) to 'one hot vector' with (batch,2^K)
def one_hot(bit_vec):
    bit_vec = bit_vec.squeeze(-1)  # (batch, K)
    N_batch = bit_vec.size(0) 
    N_bits = bit_vec.size(1)

    ind = torch.arange(0,N_bits).repeat(N_batch,1) # (batch, K)
    ind = ind.to(bit_vec.device)
    ind_vec = torch.sum( torch.mul(bit_vec,2**ind), axis=1).long() # batch
    b_onehot = torch.zeros((N_batch, 2**N_bits), dtype=int)
    for ii in range(N_batch):
        b_onehot[ii, ind_vec[ii]]=1 # one-hot vector
    return b_onehot

def dec2bin(x, N_bits):
    mask = 2**torch.arange(N_bits)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def qfunc(x):
    return .5-.5*erf(x/np.sqrt(2))

# def SER(B,SNR):
#     a = np.sqrt(3*(2**B-1)/(2**B+1))
#     d = 2*a/(2**B-1)
#     ser = (2**(B+1)-1) / 2**B*qfunc(d/2*np.sqrt(SNR))

def error_rate_bitvector(b_est,b):
    b = np.round(b)
    b_est = np.round(b_est)
    error_matrix = np.not_equal(b,b_est).float()
    N_batch = error_matrix.shape[0]
    N_bits = error_matrix.shape[1]
    ber = torch.sum(torch.sum(error_matrix)) / (N_batch*N_bits)
    bler = torch.sum((torch.sum(error_matrix,axis=1)>0))/N_batch

    return ber, bler

def error_rate_onehot(d_est, b, tot_N_bits=None):
    ind_est = torch.argmax(d_est, dim=1).squeeze(-1)
    N_batch = b.size(0)
    N_bits = b.size(1)
    b_est = dec2bin(ind_est, N_bits)
    b = b.squeeze(-1)
    if tot_N_bits is not None:
        b_est = b_est.view(-1,tot_N_bits)
        b = b.view(-1,tot_N_bits)
    
    error_matrix = np.not_equal(b, b_est).float()
    ber = torch.sum(torch.sum(error_matrix))/(N_batch*N_bits)
    bler = torch.sum((torch.sum(error_matrix,dim=1)>0))/N_batch

    return ber, bler

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
