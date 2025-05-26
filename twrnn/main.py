import math
import numpy as np
from scipy.special import erf
import sys, os, re
import time
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from params import params
from datetime import datetime

from twrnn_class import Twoway_coding
from utils import *
from test_model import test_model
from make_argparser import make_parser

if __name__=='__main__':
    # model setup
    parser, _ = make_parser()
    parameter = parser.parse_args(sys.argv[1:])
    use_cuda = parameter.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    parameter.device = device

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

    if parameter.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir = os.path.join(parameter.save_dir, f'tblog_{datetime.now().strftime("False%Y%m%d-%H%M%S")}'))

    # Generate training data
    SNR1 = parameter.SNR1               # SNR at User1 in dB
    np1 = 10**(-SNR1/10)   # noise power1 -- Assuming signal power is set to 1
    sigma1 = np.sqrt(np1)
    SNR2 = parameter.SNR2               # SNR at User2 in dB
    np2 = 10**(-SNR2/10)
    sigma2 = np.sqrt(np2)

    # Training set: tuples of (stream, noise1, noise 2)
    N_train = int(1e7)  # number of training set
    bit1_train     = torch.randint(0, 2, (N_train, parameter.N_bits, 1))
    bit2_train     = torch.randint(0, 2, (N_train, parameter.N_bits, 1))
    noise1_train   = sigma1*torch.randn((N_train, parameter.N_channel_use, 1)) 
    noise2_train   = sigma2*torch.randn((N_train, parameter.N_channel_use, 1)) 

    # Validation
    N_validation = int(1e5)

    print('np1: ', np1)
    print('np2: ', np2)

    if use_cuda:
        model = Twoway_coding(parameter).to(device)
        torch.backends.cudnn.benchmark = True
    else:
        model = Twoway_coding(parameter)

    model.sigma1 = sigma1
    model.sigma2 = sigma2

    print(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameter.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # Training
    num_epoch = 100
    clipping_value = parameter.grad_clip
    
    print('Before training ')
    print('weight_power1: ', model.weight_power1_normalized.cpu().detach().numpy().round(3))
    print('weight_power2: ', model.weight_power2_normalized.cpu().detach().numpy().round(3))
    if parameter.attention_type==4:
        print('weight1_merge: ', model.weight1_merge_normalized.cpu().detach().numpy().round(3))
        print('weight2_merge: ', model.weight2_merge_normalized.cpu().detach().numpy().round(3))
    if parameter.attention_type==5:
        print('weight1_merge_fwd: ', model.weight1_merge_normalized_fwd.cpu().detach().numpy().round(3))
        print('weight1_merge_bwd: ', model.weight1_merge_normalized_bwd.cpu().detach().numpy().round(3))
        print('weight2_merge_fwd: ', model.weight2_merge_normalized_fwd.cpu().detach().numpy().round(3))
        print('weight2_merge_bwd: ', model.weight2_merge_normalized_bwd.cpu().detach().numpy().round(3))
    print()

    N_iter = (N_train//parameter.batch_size)
    print(f'Device: {device}')
    print(f'N_iter: {N_iter}')
    nowtime_int = int(time.time())
    save_freq = num_epoch // 20

    epoch_start = 0
    ctr = 0
    if parameter.loadfile is not None:
        ckpt = torch.load(parameter.loadfile)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        epoch_start = ckpt['epoch']
        ctr = epoch_start * N_iter
    

    pbar = tqdm(total=num_epoch-epoch_start)
    total_run_starttime = datetime.now()

    for epoch in range(num_epoch):
        pbar.update(1)
        model.train() # model.training() becomes True
        loss_training = 0
        for i in range(N_iter):
            bit1 = bit1_train[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_bits,1) 
            bit2 = bit2_train[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_bits,1) 
            noise1 = noise1_train[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_channel_use,1)
            noise2 = noise2_train[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_channel_use,1)

            bit1   = bit1.to(device)
            bit2   = bit2.to(device)
            noise1 = noise1.to(device)
            noise2 = noise2.to(device)

            # forward pass
            optimizer.zero_grad() 
            X2_hat, X1_hat = model(bit1, bit2, noise1, noise2)

            # Define loss according to output type
            if parameter.output_type == 'bit_vector':
                loss = F.binary_cross_entropy(X1_hat, bit1) + F.binary_cross_entropy(X2_hat, bit2)
            elif parameter.output_type == 'one_hot_vector':
                bit1_hot =  one_hot(bit1).view(parameter.batch_size, 2**parameter.N_bits, 1) # (batch,2^K,1)
                bit2_hot =  one_hot(bit2).view(parameter.batch_size, 2**parameter.N_bits, 1)
                loss = (F.cross_entropy(X1_hat.squeeze(-1), torch.argmax(bit1_hot,dim=1).squeeze(-1).to(device)) # (batch,2^K), (batch)
                        + F.cross_entropy(X2_hat.squeeze(-1), torch.argmax(bit2_hot,dim=1).squeeze(-1).to(device)))
            
            # training
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            loss_training += loss.item()
            optimizer.step()
            
            if i % 100 == 0:
                x = int(time.time())
                print(f'Epoch: {epoch}, Iter: {i} out of {N_iter}, Loss: {loss.item():.4f}, Time: {x-nowtime_int}')
                eprint(f'Epoch: {epoch}, Iter: {i} out of {N_iter}, Loss: {loss.item():.4f}, Time: {x-nowtime_int}')
                nowtime_int = x

        if epoch % save_freq == 0:
            nowtime=datetime.now().strftime('%Y%m%d-%H%M%S')
            torch.save({'epoch' : epoch+1,
                        'model_state_dict' : model.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'scheduler_state_dict' : scheduler.state_dict(),
                        'loss' : loss.item()},
                        os.path.join(parameter.save_dir, f'{nowtime}.pt'))

        ber1_val, ber2_val, bler1_val, bler2_val, _, _ = test_model(model, parameter, N_validation)
        # Summary of each epoch
        print('Summary: Epoch: {}, lr: {}, Average loss: {:.4f}, BLER: {:.4f}'.format(epoch, optimizer.param_groups[0]['lr'], loss_training/N_iter, bler1_val+bler2_val) )
        eprint('Summary: Epoch: {}, lr: {}, Average loss: {:.4f}, BLER: {:.4f}'.format(epoch, optimizer.param_groups[0]['lr'], loss_training/N_iter, bler1_val+bler2_val) )
        if parameter.use_tensorboard:
            writer.add_scalar('BLER', bler1_val+bler2_val)
            writer.add_scalar('BLER1', bler1_val)
            writer.add_scalar('BLER2', bler2_val)

        scheduler.step() # reduce learning rate
        
        print('weight_power1: ', model.weight_power1_normalized.cpu().detach().numpy().round(3))
        print('weight_power2: ', model.weight_power2_normalized.cpu().detach().numpy().round(3))
        if parameter.attention_type==4:
            print('weight1_merge: ', model.weight1_merge_normalized.cpu().detach().numpy().round(3))
            print('weight2_merge: ', model.weight2_merge_normalized.cpu().detach().numpy().round(3))
        if parameter.attention_type==5:
            print('weight1_merge_fwd: ', model.weight1_merge_normalized_fwd.cpu().detach().numpy().round(3))
            print('weight1_merge_bwd: ', model.weight1_merge_normalized_bwd.cpu().detach().numpy().round(3))
            print('weight2_merge_fwd: ', model.weight2_merge_normalized_fwd.cpu().detach().numpy().round(3))
            print('weight2_merge_bwd: ', model.weight2_merge_normalized_bwd.cpu().detach().numpy().round(3))
        print()
        
        # Validation
        print('Ber1:  ', float(ber1_val))
        print('Ber2:  ', float(ber2_val))
        print('Bler1: ', float(bler1_val))
        print('Bler2: ', float(bler2_val))
        print()

    nowtime=datetime.now().strftime('%Y%m%d-%H%M%S')
    torch.save({'epoch' : epoch+1,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'loss' : loss.item()},
                os.path.join(parameter.save_dir, 'final.pt'))

    print(f'\nTotal runtime: {datetime.now()-total_run_starttime}')
    sys.stdout = orig_stdout
    outfile.close()