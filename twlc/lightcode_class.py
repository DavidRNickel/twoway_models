import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.init import kaiming_uniform_

import numpy as np
from math import sqrt, pi
from scipy.special import j0 #0th order Bessel function, generic softmax
import sys

from enc_dec_block import EncDecBlock

class Lightcode(nn.Module):
    def __init__(self,conf):
        super(Lightcode,self).__init__()
        
        # torch model setup
        self.conf = conf
        self.device = conf.device
        self.training = True

        # model mode
        self.is_two_way = conf.is_two_way
        self.is_one_way_active = conf.is_one_way_active
        self.is_one_way_passive = conf.is_one_way_passive

        # bitstream parameters
        self.K = conf.K # block-length
        self.M = conf.M # sub-block-length
        self.T = conf.T # number of channel uses to transmit M symbols
        print(f'K: {self.K}, M: {self.M}, T: {self.T}')

        # power and SNR
        self.noise_pwr_ff = conf.noise_pwr_ff # SNR1: noise from User 1 to User 2
        self.noise_pwr_fb = conf.noise_pwr_fb # SNR2: noise from User 2 to User 1
        print(f'SNR1: {conf.snr_ff}')
        print(f'SNR2: {conf.snr_fb}')
        self.pwr_factor_1 = torch.sqrt(torch.tensor(conf.power_factor_1))#.to(self.device)) # make !=1 for non-unit-power scale
        self.pwr_factor_2 = torch.sqrt(torch.tensor(conf.power_factor_2))#.to(self.device))

        # encoders and decoders
        self.enc_1 = EncDecBlock(conf.knowledge_vec_len, 1, conf, is_encoder=True, d_model=conf.d_model)
        if self.is_two_way:
            self.enc_2 = EncDecBlock(conf.knowledge_vec_len, 1, conf, is_encoder=True, d_model=conf.d_model)
            self.dec_1 = EncDecBlock(2*self.T+self.M, 2**self.M, conf, d_model=conf.d_model)
            self.dec_2 = EncDecBlock(2*self.T+self.M, 2**self.M, conf, d_model=conf.d_model)
        elif self.is_one_way_active:
            self.enc_2 = EncDecBlock(conf.knowledge_vec_len, 1, conf, is_encoder=True, d_model=conf.d_model)
            self.dec_1 = EncDecBlock(2*self.T, 2**self.M, conf, d_model=conf.d_model)
        else:
            self.enc_2 = lambda x: x # doing this just keeps code more consistent in transmit_symbol/process_bits_at_receiver
            self.dec_1 = EncDecBlock(self.T, 2**self.M, conf, d_model=conf.d_model)

        # Power weighting-related parameters.
        self.wgt_pwr_1 = torch.nn.Parameter(torch.Tensor(self.T), requires_grad=True)
        self.wgt_pwr_1.data.uniform_(1., 1.)
        self.wgt_pwr_normed_1 = torch.sqrt(self.wgt_pwr_1**2 * (self.T)/torch.sum(self.wgt_pwr_1**2))
        self.xmit_pwr_track_1 = []

        # Parameters for normalizing mean and variance of transmit signals.
        self.mean_batch_1 = torch.zeros(self.T)
        self.std_batch_1 = torch.ones(self.T)
        self.mean_saved_1 = torch.zeros(self.T)
        self.std_saved_1 = torch.zeros(self.T)

        if not self.is_one_way_passive:
            self.wgt_pwr_2 = torch.nn.Parameter(torch.Tensor(self.T), requires_grad=True)
            self.wgt_pwr_2.data.uniform_(1., 1.)
            self.wgt_pwr_normed_2 = torch.sqrt(self.wgt_pwr_2**2 * (self.T)/torch.sum(self.wgt_pwr_2**2))
            self.xmit_pwr_track_2 = []
        
            self.mean_batch_2 = torch.zeros(self.T)
            self.std_batch_2 = torch.ones(self.T)
            self.mean_saved_2 = torch.zeros(self.T)
            self.std_saved_2 = torch.zeros(self.T)

        self.normalization_with_saved_data = False # True: inference w/ saved mean, var; False: calculate mean, var

    #
    #
    def forward(self, bitstreams_1, bitstreams_2, noise_ff=None, noise_fb=None):
        know_vec_1 = self.make_knowledge_vec(bitstreams_1, t=0, fb_info = None, prev_x = None)
        if self.is_two_way:
            know_vec_2 = self.make_knowledge_vec(bitstreams_2, t=0, fb_info = None, prev_x = None) 
        else: 
            know_vec_2 = None

        self.wgt_pwr_normed_1 = torch.sqrt(1e-8 + self.wgt_pwr_1**2 * (self.T) / (self.wgt_pwr_1**2).sum())
        if not self.is_one_way_passive:
            self.wgt_pwr_normed_2 = torch.sqrt(1e-8 + self.wgt_pwr_2**2 * (self.T) / (self.wgt_pwr_2**2).sum())
        # print(f'\nwp1: {self.wgt_pwr_1}, wpn1: {self.wgt_pwr_normed_1}')

        if noise_ff is None:
            noise_ff = sqrt(self.noise_pwr_ff) * torch.randn((bitstreams_1.shape[0], self.T), requires_grad=False, device=self.device)
            noise_fb = sqrt(self.noise_pwr_fb) * torch.randn((bitstreams_1.shape[0], self.T), requires_grad=False, device=self.device)
        
        # used to track the transmit power of each user
        self.xmit_pwr_track_1, self.xmit_pwr_track_2 = [], []
        self.recvd_y_1, self.recvd_y_2 = None, None

        for t in range(self.T):
            # Transmit side
            x1 = self.transmit_symbol(know_vec_1, self.enc_1, t, uid=1)

            if self.is_two_way:
                x2 = self.transmit_symbol(know_vec_2, self.enc_2, t, uid=2)
            elif self.is_one_way_active: 
                # transmit symbols for active fb encoder will be made inside p_b_a_r
                x2 = bitstreams_2 
            else:
                x2 = None

            # recvd_y_i and prev_xmit_signal_i are populated in here
            self.process_bits_at_receiver(x1, x2, t, noise_ff, noise_fb)

            if t < self.T-1: # no need to do anything after the last transmission
                know_vec_1 = self.make_knowledge_vec(bitstreams_1, t=t+1, 
                                                     fb_info = self.recvd_y_1, 
                                                     prev_x = self.prev_xmit_signal_1)
                if self.is_two_way:
                    know_vec_2 = self.make_knowledge_vec(bitstreams_2, t=t+1,
                                                         fb_info = self.recvd_y_2,
                                                         prev_x = self.prev_xmit_signal_2)
                                                                
        if self.is_two_way:
            dec_out_1, dec_out_2 = self.decode_received_symbols(torch.hstack((bitstreams_2, 
                                                                              self.recvd_y_2, 
                                                                              self.prev_xmit_signal_2)), 
                                                                torch.hstack((bitstreams_1, 
                                                                              self.recvd_y_1, 
                                                                              self.prev_xmit_signal_1)))
        elif self.is_one_way_active:
            dec_out_1, dec_out_2 = self.decode_received_symbols(torch.hstack((self.recvd_y_2, self.prev_xmit_signal_2[:,1:])), self.recvd_y_1)

        else:
            # dec_out_2 will be set to None here.
            dec_out_1, dec_out_2 = self.decode_received_symbols(self.recvd_y_2, None)

        return dec_out_1, dec_out_2

    #
    #
    def make_knowledge_vec(self, b, t, fb_info=None, prev_x=None):
        bs = b.shape[0]
        # only need to check if it's active or passive (in some sense...)
        if fb_info is not None:
            fbi = torch.hstack((fb_info-prev_x, torch.zeros((bs, self.T-1-t), device=self.device)))
            px = torch.hstack((prev_x, torch.zeros((bs, self.T-1-t), device=self.device)))
        else:
            fbi = torch.zeros((bs, self.T-1), device=self.device)
            px = torch.zeros((bs, self.T-1), device=self.device)
        
        return torch.hstack((b, px, fbi))
        
        
    #
    # Process the received symbols at the decoder side. NOT THE DECODING STEP!!!
    def process_bits_at_receiver(self, x1, x2, t, noise_ff, noise_fb):
        y2 =  x1 + noise_ff[:,t]

        if t!= 0: 
            self.prev_xmit_signal_1 = torch.hstack((self.prev_xmit_signal_1, x1.unsqueeze(-1)))
            self.recvd_y_2 = torch.hstack((self.recvd_y_2, y2.unsqueeze(-1)))
        else:
            self.prev_xmit_signal_1 = x1.unsqueeze(-1)
            self.recvd_y_2 = y2.unsqueeze(-1)

        if self.is_one_way_active: # x2 is bitstreams_2 in one_way_active case
            if t==0:
                self.prev_xmit_signal_2 = torch.zeros((x1.shape[0],1), device=self.device)
            if t<self.T-1:
                know_vec = self.make_knowledge_vec(x2, t+1, fb_info = self.recvd_y_2, prev_x = self.prev_xmit_signal_2)
                x2 = self.transmit_symbol(know_vec, self.enc_2, t, uid=2)
            else:
                x2 = torch.zeros(x1.shape[0], device=self.device)

        elif self.is_one_way_passive:
            x2 = self.transmit_symbol(y2, self.enc_2, t, uid=2)

        y1 =  x2 + noise_fb[:,t]

        if t != 0:
            self.recvd_y_1 = torch.hstack((self.recvd_y_1, y1.unsqueeze(-1)))
            self.prev_xmit_signal_2 = torch.hstack((self.prev_xmit_signal_2, x2.unsqueeze(-1)))
        else:
            self.recvd_y_1 = y1.unsqueeze(-1)
            if self.is_one_way_active:
                self.prev_xmit_signal_2 = torch.hstack((self.prev_xmit_signal_2, x2.unsqueeze(-1)))
            else:
                self.prev_xmit_signal_2 = x2.unsqueeze(-1)

        if self.normalization_with_saved_data: # only tracking power during testing 
            self.xmit_pwr_track_1.append(torch.mean(torch.abs(x1)**2).detach().clone().cpu().numpy())
            if not self.is_one_way_passive:
                self.xmit_pwr_track_2.append(torch.mean(torch.abs(x2)**2).detach().clone().cpu().numpy())
        
        return 

    #
    # Actually decode all of the received symbols.
    def decode_received_symbols(self, y1, y2):
        y1 = self.dec_1(y1)
        if self.is_two_way:
            return y1, self.dec_2(y2)
        else:
            return y1, None

    #
    #
    def transmit_symbol(self, know_vec, enc, t, uid):
        return self.normalize_transmit_signal_power(enc(know_vec), t, uid)

    #
    #
    def normalize_transmit_signal_power(self, x, t, uid):
        if uid==1:
            x = self.normalization(x.squeeze(-1), t, uid)
            # print(f'ntsp x: {torch.any(torch.isnan(x))}')
            x = self.pwr_factor_1 * (self.wgt_pwr_normed_1[t] * x).squeeze(-1)
            return x
        else:
            if not self.is_one_way_passive:
                x = self.normalization(x.squeeze(-1), t, uid)
                return self.pwr_factor_2 * (self.wgt_pwr_normed_2[t] * x).squeeze(-1)
            else:
                return x.squeeze(-1)
                
    def normalization(self, inputs, t, uid):
        if not self.normalization_with_saved_data: # if not running tests
            mean_batch = torch.mean(inputs)
            std_batch = torch.std(inputs)
            if not self.training:
                if uid==1:
                    self.mean_batch_1[t] = mean_batch
                    self.std_batch_1[t] = std_batch
                else:
                    self.mean_batch_2[t] = mean_batch
                    self.std_batch_2[t] = std_batch
            return (inputs - mean_batch) / (std_batch + 1e-8)

        else:
            if uid==1: 
                return (inputs - self.mean_saved_1[t]) / (self.std_saved_1[t] + 1e-8)
            else:
                return (inputs - self.mean_saved_2[t]) / (self.std_saved_2[t] + 1e-8)

    #
    # Map the onehot representations into their binary representations.
    def one_hot_to_bits(self, onehots):
        x = torch.argmax(onehots,dim=1)
        # Adapted from https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding
        bin_representations = (((x[:,None] & (1 << torch.arange(self.M,requires_grad=False).to(self.device).flip(0)))) > 0).int()

        return bin_representations

    #
    # Take the input bitstreams and map them to their one-hot representation.
    # This function is not used anymore but is left here since it's not hurting anyone...
    def bits_to_one_hot(self, bitstreams):
        # This is a torch adaptation of https://stackoverflow.com/questions/15505514/binary-numpy-array-to-list-of-integers
        # It maps binary representations to their one-hot values by first converting the rows into 
        # the base-10 representation of the binary.
        x = (bitstreams * (1<<torch.arange(bitstreams.shape[-1]-1,-1,-1).to(self.device))).sum(1)

        return F.one_hot(x, num_classes=2**self.M)

    #
    #
    def calc_error_rates(self, bit_estimates, bits):
        if not isinstance(bits,np.ndarray):
            not_eq = torch.not_equal(bit_estimates, bits)
            ber = not_eq.float().mean()
            bler = (not_eq.sum(1)>0).float().mean()
        else:
            not_eq = np.not_equal(bit_estimates, bits)
            ber = not_eq.mean()
            bler = (not_eq.sum(1)>0).mean()

        return ber, bler
