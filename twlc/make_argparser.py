import argparse
from distutils.util import strtobool
import time
import torch

def make_parser():
    t = int(time.time())
    parser = argparse.ArgumentParser(description='Lightcode TWC Arguments')

    parser.add_argument('--use-tensorboard', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--save-freq', type=int, default=10, help='frequency (in terms of epochs) at which the model is saved throughout training')
    parser.add_argument('--loadfile', type=str, default=None, nargs='?', help='file containing saved torch model for restarting training or running tests')
    parser.add_argument('--save-dir', type=str, default=None, nargs='?', help='directory to which everything is saved', required=True)
    parser.add_argument('--log-file', type=str, default=None, nargs='?', help='file to which train/test logs are written', required=True)
    parser.add_argument('--show-progress-interval', type=int, default=None, nargs='?', help='use this if you want to print out during testing')

    parser.add_argument('--K', type=int, default=6, help='total length of bitstream')
    parser.add_argument('--M', type=int, default=3, help='length of subdivided bitstream; typically make this a factor of K')
    parser.add_argument('--T', type=int, default=9, help='number of channel uses allotted to transmitting M bits (NOT K!)')
    
    parser.add_argument('--snr-ff', type=float, default=1, help='SNR1 (NOT NECESSARILY FEED-FORWARD SNR)')
    parser.add_argument('--snr-fb', type=float, default=20, help='SNR2 (NOT NECESSARILY FEEDBACK SNR)')
    parser.add_argument('--power-factor-1', type=float, default=1, help='Multiplicative factor for scaling transmit power of user 1')
    parser.add_argument('--power-factor-2', type=float, default=1, help='Multiplicative factor for scaling transmit power of user 2')
    parser.add_argument('--is-two-way', type=lambda x: bool(strtobool(x)), default = True, nargs='?', const=True, help='two-way network')
    parser.add_argument('--is-one-way-active', type=lambda x: bool(strtobool(x)), default = False, nargs='?', const=True, help='one-way with active feedback')
    parser.add_argument('--is-one-way-passive', type=lambda x: bool(strtobool(x)), default = False, nargs='?', const=True, help='one-way with passive feedback')

    if torch.cuda.is_available(): d = 'cuda'
    elif torch.backends.mps.is_available(): d = 'mps'
    else: d = 'cpu'
    parser.add_argument('--device', type=str, default=d, nargs='?', help='set \'cuda:0\' (or \'mps\') if available else \'cpu\'')

    parser.add_argument('--optim-lr', type=float, default=.001, help='optimizer learning rate')
    parser.add_argument('--optim-weight-decay', type=float, default=.01, help='decay for AdamW optimizer')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-iters-per-epoch', type=int, default=1000)
    parser.add_argument('--num-test-epochs', type=int, default=1, help='number of epochs to run in test_model')
    parser.add_argument('--num-valid-epochs', type=int, default=1000, help='number of epochs to find mean/std in calc_saved_mean_std')
    parser.add_argument( '--batch-size', type=int, default=int(1E5))  
    parser.add_argument('--grad-clip', type=float, default=.5)
    parser.add_argument('--d-model', type=int, default=32)

    return parser, t
