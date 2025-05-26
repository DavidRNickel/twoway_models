import torch
import numpy as np

import os, sys, re
from tqdm import tqdm

from params import params
from twrnn_class import Twoway_coding
from utils import *
from test_model import test_model
from make_argparser import make_parser

if __name__=='__main__':

    parser, _ = make_parser()
    parameter = parser.parse_args(sys.argv[1:])
    use_cuda = parameter.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    parameter.device = device
    model = Twoway_coding(parameter).to(device)
    ckpt = torch.load(parameter.loadfile)
    model.load_state_dict(ckpt['model_state_dict'])


    os.makedirs(parameter.save_dir, exist_ok=True) 
    filenames = os.listdir(parameter.save_dir)
    p = re.compile(f'^{parameter.log_file}*')
    logfiles = sorted([s for s in filenames if p.match(s)])
    if len(logfiles) == 0:
        parameter.log_file = parameter.log_file + '.txt'
    elif len(logfiles) == 1:
        parameter.log_file = parameter.log_file.split('.')[0] + '_1.txt'
    else:
        lf = logfiles[-1].split('_')
        lf_name, lf_ext = '_'.join(lf[:-1]), lf[-1]# split apart the 
        n = int(lf_ext.split('.')[0])
        parameter.log_file = lf_name + f'_{int(n+1)}.txt'
    orig_stdout = sys.stdout
    outfile = open(os.path.join(parameter.save_dir, parameter.log_file), 'w')
    sys.stdout=outfile

    SNR1 = parameter.SNR1               # SNR at User1 in dB
    np1 = 10**(-SNR1/10)   # noise power1 -- Assuming signal power is set to 1
    sigma1 = np.sqrt(np1)
    SNR2 = parameter.SNR2               # SNR at User2 in dB
    np2 = 10**(-SNR2/10)
    sigma2 = np.sqrt(np2)
    model.sigma1 = sigma1
    model.sigma2 = sigma2

    # N_train = int(1e7)  # number of training set
    N_train = int(1e5)
    bit1_train     = torch.randint(0, 2, (N_train, parameter.N_bits, 1))
    bit2_train     = torch.randint(0, 2, (N_train, parameter.N_bits, 1))
    noise1_train   = sigma1*torch.randn((N_train, parameter.N_channel_use, 1)) 
    noise2_train   = sigma2*torch.randn((N_train, parameter.N_channel_use, 1)) 

    # Validation
    N_validation = int(1e5)

    # Calculate mean/var with training data
    model.eval()   # model.training() becomes False
    N_iter = N_train//parameter.batch_size
    mean1_train = torch.zeros(parameter.N_channel_use)
    std1_train  = torch.zeros(parameter.N_channel_use)
    mean2_train = torch.zeros(parameter.N_channel_use)
    std2_train  = torch.zeros(parameter.N_channel_use)
    mean1_total = 0
    std1_total = 0
    mean2_total = 0
    std2_total = 0

    with torch.no_grad():
        for i in range(N_iter):
            bit1   = bit1_train[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_bits,1) 
            bit2   = bit2_train[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_bits,1) 
            noise1 = noise1_train[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(bit1.shape[0], parameter.N_channel_use,1)
            noise2 = noise2_train[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(bit2.shape[0], parameter.N_channel_use,1)

            bit1   = bit1.to(device)
            bit2   = bit2.to(device)
            noise1 = noise1.to(device)
            noise2 = noise2.to(device)
            
            X2_hat, X1_hat = model(bit1, bit2, noise1, noise2)
            mean1_total += model.mean1_batch
            std1_total  += model.std1_batch
            mean2_total += model.mean2_batch
            std2_total  += model.std2_batch
            if i%100==0: print(i)
            
    mean1_train = mean1_total/N_iter
    std1_train = std1_total/N_iter
    mean2_train = mean2_total/N_iter
    std2_train = std2_total/N_iter
    print('Mean1: ',mean1_train)
    print('std1 : ',std1_train)
    print('Mean2: ',mean2_train)
    print('std2 : ',std2_train)

    # Inference stage
    # N_inference = int(4E8) 
    N_inference = int(1e9) 
    # N_inference = int(1e10) 
    N_small = int(1e5) # In case that N_inference is very large, we divide into small chunks
    N_iter  = N_inference//N_small

    model.normalization_with_saved_data = True
    model.mean1_saved = mean1_train
    model.std1_saved  = std1_train
    model.mean2_saved = mean2_train
    model.std2_saved  = std2_train

    ber1_sum  = 0
    bler1_sum = 0
    ber2_sum  = 0
    bler2_sum = 0
    power1_sum = np.zeros((parameter.batch_size, parameter.N_channel_use ,1))
    power2_sum = np.zeros((parameter.batch_size, parameter.N_channel_use ,1))

    pbar = tqdm(total=N_iter)
    for ii in range(N_iter):
        pbar.update(1)
        ber1_tmp, ber2_tmp, bler1_tmp, bler2_tmp, power1_tmp, power2_tmp = test_model(model, parameter, N_small)
        ber1_sum += ber1_tmp
        ber2_sum += ber2_tmp
        bler1_sum += bler1_tmp
        bler2_sum += bler2_tmp
        power1_sum += power1_tmp # (batch, N, 1)
        power2_sum += power2_tmp
        # if ii%100==0: 
        #     print('Iter: {} out of {}'.format(ii, N_iter))
        #     print('Ber1:  ', float(ber1_sum/(ii+1)))
        #     print('Ber2:  ', float(ber2_sum/(ii+1)))
        #     print('Bler1: ', float(bler1_sum/(ii+1)))
        #     print('Bler2: ', float(bler2_sum/(ii+1)))
        #     print('Power1: ', round(np.sum(power1_sum)/(parameter.batch_size*(ii+1)),3))
        #     print('Power2: ', round(np.sum(power2_sum)/(parameter.batch_size*(ii+1)),3))

    ber1_inference  = ber1_sum/N_iter
    ber2_inference  = ber2_sum/N_iter
    bler1_inference = bler1_sum/N_iter
    bler2_inference = bler2_sum/N_iter

    print()
    print('Ber1:  ', float(ber1_inference))
    print('Ber2:  ', float(ber2_inference))
    print('Bler1: ', float(bler1_inference))
    print('Bler2: ', float(bler2_inference))
    print('Bler total: ', float(bler1_inference) + float(bler2_inference))
    print('Power1: ', round(np.sum(power1_sum)/(parameter.batch_size*N_iter),3))
    print('Power2: ', round(np.sum(power2_sum)/(parameter.batch_size*N_iter),3))

    ######## Save model

    parameter.save_dir = parameter.save_dir
    torch.save(model.state_dict(), os.path.join(parameter.save_dir,'final_state_dict.pt'))

    ####### Save normalization weights
    torch.save(mean1_train, os.path.join(parameter.save_dir,'mean1_train.pt'))
    torch.save(std1_train, os.path.join(parameter.save_dir,'std1_train.pt'))
    torch.save(mean2_train, os.path.join(parameter.save_dir,'mean2_train.pt'))
    torch.save(std2_train, os.path.join(parameter.save_dir,'std2_train.pt'))
