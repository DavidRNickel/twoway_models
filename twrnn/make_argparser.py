import argparse
from distutils.util import strtobool
import time
import torch

def make_parser():
    t = int(time.time())
    parser = argparse.ArgumentParser(description='TWRNN Arguments')

    parser.add_argument('--use-tensorboard', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use-cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--save-freq', type=int, default=10, help='frequency (in terms of epochs) at which the model is saved throughout training')
    parser.add_argument('--loadfile', type=str, default=None, nargs='?', help='file containing saved torch model for restarting training or running tests')
    parser.add_argument('--save-dir', type=str, default='tmp', nargs='?', help='directory to which everything is saved')
    parser.add_argument('--train-log-file', type=str, default='tmptrain', nargs='?', help='file to which train logs are written')
    parser.add_argument('--test-log-file', type=str, default='tmptest', nargs='?', help='file to which test logs are written')
    parser.add_argument('--show-progress-interval', type=int, default=None, nargs='?', help='use this if you want to print out during testing')

    # Encoder
    parser.add_argument('--encoder-act-func', type=str, default='tanh', nargs='?', help='')
    parser.add_argument('--encoder-N-layers', type=int, default=2, help='number of RNN layers at decoder')
    parser.add_argument('--encoder-N-neurons', type=int, default=50, help='number of neurons at each RNN')

    # Decoder
    parser.add_argument('--decoder-N-layers', type=int, default=2, help='number of RNN layers at decoder')
    parser.add_argument('--decoder-N-neurons', type=int, default=50, help='number of neurons at each RNN')
    parser.add_argument('--decoder-bidirection', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--attention-type', type=int, default=5, help='see make_argparser.py for explanation')
    # 1. Only the last timestep (N-th)
    # 2. Merge the last outputs of forward/backward RNN
    # 3. Sum over all timesteps
    # 4. Attention mechanism with N weights (same weight for forward/backward)
    # 5. Attention mechanism with 2N weights (separate weights for forward/backward)

    # Channel coding information
    parser.add_argument('--tot-N-bits', type=int, default=6, help='K: total length of bitstream')
    parser.add_argument('--N-bits', type=int, default=3, help='M: length of sub-block; typically make this a factor of K')
    parser.add_argument('--N-channel-use', type=int, default=9, help='T: number of channel uses allotted to transmitting M bits (NOT K!)')
    parser.add_argument('--input-type', type=str, default='bit_vector', nargs='?', help='one of "bit_vector" or "one_hot_vector"')
    parser.add_argument('--output-type', type=str, default='one_hot_vector', nargs='?', help='one of "bit_vector" or "one_hot_vector"')
    parser.add_argument('--decoder-info', type=str, default='None', nargs='?', help='one of "bit_estimate", "state_vector", "None" for encoder input')
    parser.add_argument('--encoder-info', type=str, default='tran_symbol', nargs='?', help='one of "tran_symbol", "state_vector", "None" for decoder input')
    parser.add_argument('--SNR1', type=float, default=1, help='SNR1') 
    parser.add_argument('--SNR2', type=float, default=20, help='SNR2')


    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=.01, help='optimizer learning rate')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-iters-per-epoch', type=int, default=1000)
    parser.add_argument('--num-test-epochs', type=int, default=1, help='number of epochs to run in test_model')
    parser.add_argument('--num-valid-epochs', type=int, default=1000, help='number of epochs to find mean/std in calc_saved_mean_std')
    parser.add_argument( '--batch-size', type=int, default=int(2.5e4))  
    parser.add_argument('--grad-clip', type=float, default=1)
    parser.add_argument('--d-model', type=int, default=16)

    return parser, t
