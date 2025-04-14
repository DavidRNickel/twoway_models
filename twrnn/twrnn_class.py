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


class Twoway_coding(torch.nn.Module):
    def __init__(self, param):
        super(Twoway_coding, self).__init__()
        
        # import parameter
        self.param = param
        self.device = param.device
        if self.param.decoder_bidirection == True:
            self.decoder_bi = 2 # bi-direction
        else:
            self.decoder_bi = 1 # uni-direction

        ### Encoder input type
        # 1. input_type (bit vector, one-hot vector) for Encoder
        if self.param.input_type == 'bit_vector':
            self.num_input = self.param.N_bits
        elif self.param.input_type == 'one_hot_vector':
            self.num_input = 2**self.param.N_bits
        
        # 2. Decoder Info (bit estimate, state vector) for Encoder
        if self.param.decoder_info == 'bit_estimate':
            self.num_D = self.param.N_bits
        elif self.param.decoder_info == 'state_vector':
            self.num_D = self.decoder_bi * self.param.decoder_N_neurons # 2*50 = 100
        elif self.param.decoder_info == 'None':
            self.num_D = 0
        
        ### Decoder input type
        # 1. output_type (bits, one-hot vector) for Decoder
        if self.param.output_type == 'bit_vector':
            self.num_output = self.param.N_bits
        elif self.param.output_type == 'one_hot_vector':
            self.num_output = 2**self.param.N_bits
            
        # 2. Encoder Info ('tran_symbol', 'state_vector', 'None') for Decoder
        if self.param.encoder_info == 'tran_symbol':
            self.num_E = 1
        elif self.param.encoder_info == 'state_vector':
            self.num_E = self.param.encoder_N_neurons # 50
        elif self.param.encoder_info == 'None':
            self.num_E = 0
            

        # encoder 1. RNN
        self.encoder1_RNN   = torch.nn.GRU(self.num_input + 1 + self.num_D, self.param.encoder_N_neurons, num_layers = self.param.encoder_N_layers, 
                                          bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.encoder1_linear = torch.nn.Linear(self.param.encoder_N_neurons, 1)
        
        # encoder 2. RNN
        self.encoder2_RNN   = torch.nn.GRU(self.num_input + 1 + self.num_D, self.param.encoder_N_neurons, num_layers = self.param.encoder_N_layers, 
                                          bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.encoder2_linear = torch.nn.Linear(self.param.encoder_N_neurons, 1)

        # power weight 1
        self.weight_power1 = torch.nn.Parameter(torch.Tensor(self.param.N_channel_use), requires_grad = True )
        self.weight_power1.data.uniform_(1.0, 1.0) # all 1
        self.weight_power1_normalized = torch.sqrt(self.weight_power1**2 *(self.param.N_channel_use)/torch.sum(self.weight_power1**2))
        
        # power weight 2
        self.weight_power2 = torch.nn.Parameter(torch.Tensor(self.param.N_channel_use), requires_grad = True )
        self.weight_power2.data.uniform_(1.0, 1.0) # all 1
        self.weight_power2_normalized = torch.sqrt(self.weight_power2**2 *(self.param.N_channel_use)/torch.sum(self.weight_power2**2))
        
        # decoder 1
        self.decoder1_RNN = torch.nn.GRU(self.num_input + 1 + self.num_E, self.param.decoder_N_neurons, num_layers = self.param.decoder_N_layers, 
                                        bias=True, batch_first=True, dropout=0, bidirectional= self.param.decoder_bidirection) 
        self.decoder1_linear = torch.nn.Linear(self.decoder_bi*self.param.decoder_N_neurons, self.num_output) # 100,10
        
        # decoder 2
        self.decoder2_RNN = torch.nn.GRU(self.num_input + 1 + self.num_E, self.param.decoder_N_neurons, num_layers = self.param.decoder_N_layers, 
                                        bias=True, batch_first=True, dropout=0, bidirectional= self.param.decoder_bidirection) 
        self.decoder2_linear = torch.nn.Linear(self.decoder_bi*self.param.decoder_N_neurons, self.num_output) # 100,10

        
        # attention type
        if self.param.attention_type==5:  # bi-directional --> 2N weights
            self.weight1_merge = torch.nn.Parameter(torch.Tensor(self.param.N_channel_use,2), requires_grad = True ) 
            self.weight1_merge.data.uniform_(1.0, 1.0) # all 1
            # Normalization
            self.weight1_merge_normalized_fwd = torch.sqrt(self.weight1_merge[:,0]**2 *(self.param.N_channel_use)/torch.sum(self.weight1_merge[:,0]**2)) 
            self.weight1_merge_normalized_bwd  = torch.sqrt(self.weight1_merge[:,1]**2 *(self.param.N_channel_use)/torch.sum(self.weight1_merge[:,1]**2))
        
            self.weight2_merge = torch.nn.Parameter(torch.Tensor(self.param.N_channel_use,2), requires_grad = True ) 
            self.weight2_merge.data.uniform_(1.0, 1.0) # all 1
            # Normalization
            self.weight2_merge_normalized_fwd = torch.sqrt(self.weight2_merge[:,0]**2 *(self.param.N_channel_use)/torch.sum(self.weight2_merge[:,0]**2)) 
            self.weight2_merge_normalized_bwd  = torch.sqrt(self.weight2_merge[:,1]**2 *(self.param.N_channel_use)/torch.sum(self.weight2_merge[:,1]**2))
        
        if self.param.attention_type== 4: # uni-directional --> N weights
            self.weight1_merge = torch.nn.Parameter(torch.Tensor(self.param.N_channel_use),requires_grad = True )
            self.weight1_merge.data.uniform_(1.0, 1.0) # all 1
            # Normalization
            self.weight1_merge_normalized  = torch.sqrt(self.weight1_merge**2 *(self.param.N_channel_use)/torch.sum(self.weight1_merge**2))
        
            self.weight2_merge = torch.nn.Parameter(torch.Tensor(self.param.N_channel_use),requires_grad = True )
            self.weight2_merge.data.uniform_(1.0, 1.0) # all 1
            # Normalization
            self.weight2_merge_normalized  = torch.sqrt(self.weight2_merge**2 *(self.param.N_channel_use)/torch.sum(self.weight2_merge**2))
        
        
        # Parameters for normalization (mean and variance)
        # User 1
        self.mean1_batch = torch.zeros(self.param.N_channel_use) 
        self.std1_batch = torch.ones(self.param.N_channel_use)
        self.mean1_saved = torch.zeros(self.param.N_channel_use)
        self.std1_saved = torch.ones(self.param.N_channel_use)
        # User 2
        self.mean2_batch = torch.zeros(self.param.N_channel_use) 
        self.std2_batch = torch.ones(self.param.N_channel_use)
        self.mean2_saved = torch.zeros(self.param.N_channel_use)
        self.std2_saved = torch.ones(self.param.N_channel_use)
        self.normalization_with_saved_data = False   # True: inference with saved mean/var, False: calculate mean/var

    def decoder_activation(self, inputs):
        if self.param.output_type == 'bit_vector':
            return torch.sigmoid(inputs) # training with binary cross entropy
        elif self.param.output_type == 'one_hot_vector':
            return inputs # Note. softmax function is applied in "F.cross_entropy" function
    
    # Convert `bit vector' to 'one-hot vector'
    def one_hot(self, bit_vec):
        bit_vec = bit_vec.view(self.param.batch_size, self.param.N_bits)
        N_batch = bit_vec.size(0) # batch_size
        N_bits = bit_vec.size(1)  # N_bits=K

        ind = torch.arange(0,N_bits).repeat(N_batch,1) 
        ind = ind.to(self.device)
        ind_vec = torch.sum( torch.mul(bit_vec, 2**ind), axis=1 ).long()
        bit_onehot = torch.zeros((N_batch, 2**N_bits), dtype=int)
        for ii in range(N_batch):
            bit_onehot[ii, ind_vec[ii]]=1 # one-hot vector
        return bit_onehot 
        
    def normalization(self, inputs, t_idx, user_idx):
        if self.training: # During training
            mean_batch = torch.mean(inputs)
            std_batch  = torch.std(inputs)
            outputs   = (inputs - mean_batch)/std_batch
        else: 
            if self.normalization_with_saved_data: # During inference
                if user_idx==1:
                    outputs   = (inputs - self.mean1_saved[t_idx])/self.std1_saved[t_idx]
                elif user_idx==2:
                    outputs   = (inputs - self.mean2_saved[t_idx])/self.std2_saved[t_idx]
            else: 
                # During validation
                mean_batch = torch.mean(inputs)
                std_batch  = torch.std(inputs)
                outputs   = (inputs - mean_batch)/std_batch
                # calculate mean/var after training
                if user_idx==1:
                    self.mean1_batch[t_idx] = mean_batch
                    self.std1_batch[t_idx] = std_batch
                elif user_idx==2:
                    self.mean2_batch[t_idx] = mean_batch
                    self.std2_batch[t_idx] = std_batch
        return outputs


    def forward(self, b1, b2, noise1, noise2):

        # Normalize power weights
        self.weight_power1_normalized  = torch.sqrt(self.weight_power1**2 *(self.param.N_channel_use)/torch.sum(self.weight_power1**2))
        self.weight_power2_normalized  = torch.sqrt(self.weight_power2**2 *(self.param.N_channel_use)/torch.sum(self.weight_power2**2))
        
        # Encoder input
        if self.param.input_type == 'bit_vector':
            I1 = b1 
            I2 = b2 
        elif self.param.input_type == 'one_hot_vector':
            I1 = self.one_hot(b1).to(self.device)
            I2 = self.one_hot(b2).to(self.device)
        
        for t in range(self.param.N_channel_use): # timesteps
            # Encoder
            if t == 0: # 1st timestep
                input1_total        = torch.cat([I1.view(self.param.batch_size, 1, self.num_input), 
                                               torch.zeros((self.param.batch_size, 1, self.num_D+1)).to(self.device)], dim=2) 
                ### input1_total   -- (batch, 1, num_input + num_D + 1) 
                x1_t_after_RNN, s1_t_hidden  = self.encoder1_RNN(input1_total)
                ### x1_t_after_RNN -- (batch, 1, hidden)
                ### s1_t_hidden    -- (layers, batch, hidden)
                x1_t_tilde =   self.encoder1_linear(x1_t_after_RNN)
                x1_t_tilde =   torch.tanh(self.encoder1_linear(x1_t_after_RNN))   
                
                input2_total        = torch.cat([I2.view(self.param.batch_size, 1, self.num_input), 
                                               torch.zeros((self.param.batch_size, 1, self.num_D+1)).to(self.device)], dim=2) 
                x2_t_after_RNN, s2_t_hidden  = self.encoder2_RNN(input2_total)
                x2_t_tilde =   torch.tanh(self.encoder2_linear(x2_t_after_RNN))   
                
                
            else: # 2nd-Nth timestep
                if self.param.decoder_info == 'None':
                    input1_total        = torch.cat([I1.view(self.param.batch_size, 1, self.num_input), y1_t], dim=2) 
                    input2_total        = torch.cat([I2.view(self.param.batch_size, 1, self.num_input), y2_t], dim=2) 
                else:
                    input1_total        = torch.cat([I1.view(self.param.batch_size, 1, self.num_input), y1_t, D1_tmp], dim=2) 
                    input2_total        = torch.cat([I2.view(self.param.batch_size, 1, self.num_input), y2_t, D2_tmp], dim=2)
                
                x1_t_after_RNN, s1_t_hidden  = self.encoder1_RNN(input1_total, s1_t_hidden)
                x1_t_tilde =   torch.tanh(self.encoder1_linear(x1_t_after_RNN))
                
                
                x2_t_after_RNN, s2_t_hidden  = self.encoder2_RNN(input2_total, s2_t_hidden)
                x2_t_tilde =   torch.tanh(self.encoder2_linear(x2_t_after_RNN))
            
            # Power control layer: 1. Normalization, 2. Power allocation
            x1_t_norm = self.normalization(x1_t_tilde, t, 1).view(self.param.batch_size, 1, 1)
            x1_t  = x1_t_norm * self.weight_power1_normalized[t] 
            x2_t_norm = self.normalization(x2_t_tilde, t, 2).view(self.param.batch_size, 1, 1)
            x2_t  = x2_t_norm * self.weight_power2_normalized[t] 
            
            # Forward transmission (from User 1 to 2)
            y2_t = x1_t + noise1[:,t,:].view(self.param.batch_size, 1, 1)
            
            # Backward transmission (from User 2 to 1)
            y1_t = x2_t + noise2[:,t,:].view(self.param.batch_size, 1, 1)
            
            # Concatenate values along time t
            if t == 0:
#                 x1_norm_total = x1_t_norm
                x1_total = x1_t
                x2_total = x2_t
                y1_total = y1_t
                y2_total = y2_t
            else:
#                 x_norm_total = torch.cat([x_norm_total, x_t_norm], dim=1) 
                x1_total = torch.cat([x1_total, x1_t ], dim = 1) # In the end, (batch, N, 1)
                x2_total = torch.cat([x2_total, x2_t ], dim = 1)
                y1_total = torch.cat([y1_total, y1_t ], dim = 1) 
                y2_total = torch.cat([y2_total, y2_t ], dim = 1) 
            
            # Encoder info updates
            if self.param.encoder_info == 'tran_symbol':
                E1_tmp = x1_t # (batch,1,1)
                E2_tmp = x2_t # (batch,1,1)
            elif self.param.encoder_info == 'state_vector':
                E1_tmp = x1_t_after_RNN # (batch, 1, hidden) 
                E2_tmp = x2_t_after_RNN # (batch, 1, hidden) 
#                 E1_tmp = s1_t_hidden[-1].view(self.param.batch_size, 1, -1) # (batch, 1, hidden) # only last layer
#                 E2_tmp = s2_t_hidden[-1].view(self.param.batch_size, 1, -1) # (batch, 1, hidden)
            elif self.param.encoder_info == 'None':
                E1_tmp = None
                E2_tmp = None
            
            ########################################
            # Immediate Decoding
            if self.param.decoder_info != 'None': 
                # Encoder uses decoder info
                # --> There is connection from decoder to encoder
                # --> Immediate decoding is needed!
                if self.param.encoder_info == 'None': # Decoder does not use encoder info
                    decoder_input1 = torch.cat([I1.view(self.param.batch_size, 1, self.num_input), y1_t], dim=2) # (batch, 1, num_input + 1)
                    decoder_input2 = torch.cat([I2.view(self.param.batch_size, 1, self.num_input), y2_t], dim=2) # (batch, 1, num_input + 1)
                else:
                    decoder_input1 = torch.cat([I1.view(self.param.batch_size, 1, self.num_input), y1_t, E1_tmp], dim=2) # (batch, 1, num_input + 1 + num_E)
                    decoder_input2 = torch.cat([I2.view(self.param.batch_size, 1, self.num_input), y2_t, E2_tmp], dim=2)
    
                ### decoder_input1: (batch, 1, num_input + 1 + num_E) -- batch, input seq, input size
                r1_t_last, r1_t_hidden  = self.decoder1_RNN(decoder_input1)
                ### r1_t_last   -- (batch, 1(=input seq), hidden)
                ### r1_t_hidden -- (layer, batch, hidden)
                r2_t_last, r2_t_hidden  = self.decoder2_RNN(decoder_input2)
                
                if self.param.decoder_info == 'state_vector': # No output calculation required
                    D1_tmp = r1_t_last
                    D2_tmp = r2_t_last
                    if t==0:
                        r1_hidden = r1_t_last
                        r2_hidden = r2_t_last
                    else:
                        r1_hidden = torch.cat([r1_hidden, r1_t_last], dim=1) # Finally, (batch, N, hidden)
                        r2_hidden = torch.cat([r2_hidden, r2_t_last], dim=1)
                        
#                 if self.decoder_info == 'bit_estimate':
                
#                     output1     = self.decoder_activation(self.decoder1_linear(z1_t_after_RNN)) # (batch,1,num_output)
#                     output1_last = output1.view(self.param.batch_size,-1,1) # (batch, num_output, 1)
                
        # Decoder do inference after N transmission are conducted!   
        if self.param.decoder_info == 'state_vector':
            # Only Uni-directinoal is possible
            
            # Normalize attention weights (Uni-directional attention weights)
            self.weight1_merge_normalized  = torch.sqrt(self.weight1_merge**2 *(self.param.N_channel_use)/torch.sum(self.weight1_merge**2)) 
            self.weight2_merge_normalized  = torch.sqrt(self.weight2_merge**2 *(self.param.N_channel_use)/torch.sum(self.weight2_merge**2)) 
            
            # Multiply attention weights
            r1_merge = torch.tensordot(r1_hidden, self.weight1_merge_normalized, dims=([1], [0])) # (batch, hidden)
            output1 = self.decoder_activation(self.decoder1_linear(r1_merge)) 
            output1_last = output1.view(self.param.batch_size,-1,1) # (batch, num_output, 1)

            r2_merge = torch.tensordot(r2_hidden, self.weight2_merge_normalized, dims=([1], [0])) # (batch, hidden)
            output2 = self.decoder_activation(self.decoder2_linear(r2_merge)) 
            output2_last = output2.view(self.param.batch_size,-1,1) # (batch, num_output, 1)
    
    
        # Non-immediate Decoding
        if self.param.decoder_info == 'None': # No connection from decoder to encoder
            # Normalize attention weights
            if self.param.attention_type== 4:
                self.weight1_merge_normalized  = torch.sqrt(self.weight1_merge**2 *(self.param.N_channel_use)/torch.sum(self.weight1_merge**2)) 
                self.weight2_merge_normalized  = torch.sqrt(self.weight2_merge**2 *(self.param.N_channel_use)/torch.sum(self.weight2_merge**2)) 
            if self.param.attention_type== 5:
                self.weight1_merge_normalized_fwd  = torch.sqrt(self.weight1_merge[:,0]**2 *(self.param.N_channel_use)/torch.sum(self.weight1_merge[:,0]**2)) # 30
                self.weight1_merge_normalized_bwd  = torch.sqrt(self.weight1_merge[:,1]**2 *(self.param.N_channel_use)/torch.sum(self.weight1_merge[:,1]**2))
                self.weight2_merge_normalized_fwd  = torch.sqrt(self.weight2_merge[:,0]**2 *(self.param.N_channel_use)/torch.sum(self.weight2_merge[:,0]**2)) # 30
                self.weight2_merge_normalized_bwd  = torch.sqrt(self.weight2_merge[:,1]**2 *(self.param.N_channel_use)/torch.sum(self.weight2_merge[:,1]**2))

                
            I1_tmp = I1.view(self.param.batch_size, 1, self.num_input)
            I1_copy = I1_tmp.repeat(1, self.param.N_channel_use, 1) # (batch, N, K)
            I2_tmp = I2.view(self.param.batch_size, 1, self.num_input)
            I2_copy = I2_tmp.repeat(1, self.param.N_channel_use, 1) # (batch, N, K)
            if self.param.encoder_info == 'None':
                decoder1_input = torch.cat([I1_copy, y1_total], dim=2) # (batch, N, K+1)
                decoder2_input = torch.cat([I2_copy, y2_total], dim=2) # (batch, N, K+1)
            elif self.param.encoder_info == 'tran_symbol':
                decoder1_input = torch.cat([I1_copy, x1_total, y1_total], dim=2) # (batch, N, K+2)
                decoder2_input = torch.cat([I2_copy, x2_total, y2_total], dim=2) # (batch, N, K+2)
            
            r1_hidden, _  = self.decoder1_RNN(decoder1_input) # (batch, N, bi*hidden_size)
            r2_hidden, _  = self.decoder2_RNN(decoder2_input) # (batch, N, bi*hidden_size)


    #         # Option 1. Only the N-th timestep
    #         if parameter.attention_type== 1:
    #             output     = self.decoder_activation(self.decoder_linear(r_hidden)) #(batch,N,bi*hidden)-->(batch,N,num_output)
    #             output_last = output[:,-1,:].view(self.param.batch_size,-1,1) # (batch,num_output,1)

    #         # Option 2. Merge the "last" outputs of forward/backward RNN
    #         if parameter.attention_type== 2:
    #             r_backward = r_hidden[:,0,self.param.decoder_N_neurons:] # Output at the 1st timestep of reverse RNN 
    #             r_forward = r_hidden[:,-1,:self.param.decoder_N_neurons] # Output at the N-th timestep of forward RNN
    #             r_concat = torch.cat([r_backward, r_forward ], dim = 1) 
    #             output = self.decoder_activation(self.decoder_linear(r_concat)) # (batch,num_output)
    #             output_last = output.view(self.param.batch_size,-1,1) # (batch,num_output,1)

    #         # Option 3. Sum over all timesteps
    #         if parameter.attention_type== 3:
    #             output     = self.decoder_activation(self.decoder_linear(r_hidden)) 
    #             output_last = torch.sum(output, dim=1).view(self.param.batch_size,-1,1) # (batch,num_output,1)

            # Option 4. Attention mechanism (N weights)
            if self.param.attention_type== 4:
                r1_concat = torch.tensordot(r1_hidden, self.weight1_merge_normalized, dims=([1], [0])) # (batch, hidden_size)
                output1 = self.decoder_activation(self.decoder1_linear(r1_concat)) 
                output1_last = output1.view(self.param.batch_size,-1,1) # (batch,num_output,1)
                
                r2_concat = torch.tensordot(r2_hidden, self.weight2_merge_normalized, dims=([1], [0])) # (batch, hidden_size)
                output2 = self.decoder_activation(self.decoder2_linear(r2_concat)) 
                output2_last = output2.view(self.param.batch_size,-1,1) # (batch,num_output,1)

            # Option 5. Attention mechanism (2N weights) for forward/backward
            if self.param.attention_type== 5:
                r1_hidden_forward = r1_hidden[:,:,:self.param.decoder_N_neurons]  # (batch,num_output,hidden_size)
                r1_hidden_backward = r1_hidden[:,:,self.param.decoder_N_neurons:] # (batch,num_output,hidden_size)
                r1_forward_weighted_sum = torch.tensordot(r1_hidden_forward, self.weight1_merge_normalized_fwd, dims=([1], [0]))  # (batch,hidden_size)
                r1_backward_weighted_sum = torch.tensordot(r1_hidden_backward, self.weight1_merge_normalized_bwd, dims=([1], [0]))         # (batch,hidden_size)
                r1_concat = torch.cat([r1_forward_weighted_sum, r1_backward_weighted_sum], dim = 1) 
                output1 = self.decoder_activation(self.decoder1_linear(r1_concat)) 
                output1_last = output1.view(self.param.batch_size,-1,1) # (batch,num_output,1)

                r2_hidden_forward = r2_hidden[:,:,:self.param.decoder_N_neurons]  # (batch,num_output,hidden_size)
                r2_hidden_backward = r2_hidden[:,:,self.param.decoder_N_neurons:] # (batch,num_output,hidden_size)
                r2_forward_weighted_sum = torch.tensordot(r2_hidden_forward, self.weight2_merge_normalized_fwd, dims=([1], [0]))  # (batch,hidden_size)
                r2_backward_weighted_sum = torch.tensordot(r2_hidden_backward, self.weight2_merge_normalized_bwd, dims=([1], [0]))         # (batch,hidden_size)
                r2_concat = torch.cat([r2_forward_weighted_sum, r2_backward_weighted_sum], dim = 1) 
                output2 = self.decoder_activation(self.decoder2_linear(r2_concat)) 
                output2_last = output2.view(self.param.batch_size,-1,1) # (batch,num_output,1)

            
        self.x1 = x1_total                    # (batch,N,1)
        self.x2 = x2_total                    # (batch,N,1)
        
        return output1_last, output2_last