# -*- coding: utf-8 -*-
#
# main() for CL-scheme implementation
#

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from tqdm import tqdm

from utils import calc_gamma_opt, calc_F, calc_q, calc_beta_opt, to_db


#
# Main functionality of inner code of CL scheme. Wrap in LDPC or TurboCode
# to get the full CL scheme.
def inner_code(symbols, s_2, n, snr1_linear, gg, printout_snr=False):
    rho = snr1_linear / n
    gamma_opt = calc_gamma_opt(s_2, rho, n)
    # beta_opt = np.sqrt((n - 1) / (n+(1+s_2)*n*gamma_opt*rho)) # good approximation
    beta_opt = calc_beta_opt(s_2, n, gamma_opt, rho) # actual optimal solution
    F = calc_F(s_2, beta_opt, n)
    q = calc_q(gg, F, s_2, n)

    fwd_noise = np.random.normal(0, 1, (symbols.shape[0], n, 1))
    fbk_noise = np.random.normal(0, sqrt(s_2), (symbols.shape[0], n, 1))
    xmit_symbols = F@(fwd_noise + fbk_noise) + symbols*gg
    y = xmit_symbols + fwd_noise

    if printout_snr:
        print(f'Forward SNR [dB]: {to_db(np.mean(xmit_symbols.flatten()**2)) - to_db(np.mean(fwd_noise**2))}')
        print(f'Feedback SNR [dB]: {to_db(np.mean(y.flatten()**2)) - to_db(np.mean(fbk_noise.flatten()**2))}')

    return (q.T@y).reshape(-1,1)

#
# Minimum distance decoding.
def decode_pam(recvd_syms, constel):
    return (np.abs(recvd_syms - constel)).argmin(axis=1)


if __name__=='__main__':
    num_datapoints = 1000000

    T = 9 # total number of channel uses for user
    L = 6 # total number of bits to send over T uses
    N = 3 # number of channel uses per symbol 
    modulation_order = 2

    assert L*N/modulation_order == T # make sure you're using the right number of channel uses
    assert L % modulation_order == 0 # make sure you're accounting for all symbols 
    
    M = 2**modulation_order # number of symbols in constellation
    snr1_db = 5
    snr2_db = 5
    snr1_linear = 10**(snr1_db/10)
    snr2_linear = 10**(snr2_db/10)
    
    # make 2^(mod_order) PAM symbol set
    A = np.sqrt((3*snr1_linear) / (M**2 - 1))
    sigma_2 = (1+snr1_linear) / snr2_linear #
    g = np.ones((N, 1))#/sqrt(N)
    # I'm sure there's a programmatic way to do this. I just got lazy.
    if modulation_order == 1:
        constellation = np.array([-A, A])
    elif modulation_order == 2:
        constellation = np.array([-3*A, -A, A, 3*A])
    elif modulation_order == 3:
        constellation = np.array([-7*A, -5*A, -3*A, -A, A, 3*A, 5*A, 7*A])
    elif modulation_order == 4:
        constellation = np.array([-15*A, -13*A, -11*A, -9*A, -7*A, -5*A, -3*A, -A, A, 3*A, 5*A, 7*A, 9*A, 11*A, 13*A, 15*A])
    print(f'Constellation: {constellation}')

    show_hist = False
    rounds = 10
    blers = []
    for _ in tqdm(range(rounds)):
        data = np.random.choice(np.arange(M), size=(num_datapoints, L//modulation_order))
        modulated_data = np.array([constellation[i] for i in data.flatten()]).reshape(-1,1,1)
        received_data = inner_code(modulated_data, sigma_2, N, snr1_linear, g)
        if show_hist:
            plt.figure()
            plt.hist(received_data.flatten(), bins=400)
            plt.show()
            print(f'BLER (per user): {np.mean((decoded_syms != data).sum(axis=1))}')
        decoded_syms = decode_pam(received_data, constellation).reshape(num_datapoints, L//modulation_order)
        blers.append((decoded_syms != data).sum(axis=1))
    bler = np.mean(blers)
    print(f'BLER (per-user): {bler}')
    print(f'Sum-BLER (two-way setting): {2*bler}')