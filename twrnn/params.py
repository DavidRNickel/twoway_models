import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
import time

class params():
    def __init__(self):

        # Encoder
        self.encoder_act_func = 'tanh'
        self.encoder_N_layers: int = 2    # number of RNN layers at encoder
        self.encoder_N_neurons: int = 50  # number of neurons at each RNN

        # Decoder
        self.decoder_N_layers: int = 2    # number of RNN layers at decoder
        self.decoder_N_neurons: int = 50  # number of neurons at each RNN
        self.decoder_bidirection = True   # True: bi-directional decoding, False: uni-directional decoding
        self.attention_type: int = 5      # choose the attention type among five options
        # 1. Only the last timestep (N-th)
        # 2. Merge the last outputs of forward/backward RNN
        # 3. Sum over all timesteps
        # 4. Attention mechanism with N weights (same weight for forward/backward)
        # 5. Attention mechanism with 2N weights (separate weights for forward/backward)
        
        # Setup
        self.N_bits: int = 3                # length of sub-block (M)
        self.tot_N_bits = 6                 # length of block (K)
        self.N_channel_use = 9              # number of channel uses
        self.input_type = 'bit_vector'      # choose 'bit_vector' or 'one_hot_vector'
        self.output_type = 'one_hot_vector' # choose 'bit_vector' or 'one_hot_vector'
        self.decoder_info = 'None'          # 'bit_estimate', 'state_vector', 'None' for encoder input
        self.encoder_info = 'tran_symbol'   # 'tran_symbol', 'state_vector', 'None' for decoder input

        # Learning parameters
        self.batch_size = int(2.5e4) 
        # self.batch_size = int(1E5)
        self.learning_rate = 0.01 
        self.use_cuda = True

        self.SNR1 = 1
        self.SNR2 = 30
        # self.save_results_to = f'm3t9/ff_1/fb_{int(self.SNR2)}/'
        self.save_results_to = 'sensitivity/fix_snr1_1_snr2_15/snr2_30'
        self.train_log_file = 'train_log.txt'
        self.test_log_file = 'test_log.txt'
        self.loadfile = 'm3t9/ff_1/fb_15/20250308-001548.pt'
        self.use_tensorboard = False
