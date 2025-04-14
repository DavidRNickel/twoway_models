import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import datetime
import sys
import os
import pickle as pkl

from lightcode_class import Lightcode
from test_model import test_model
from timer_class import Timer
from make_argparser import make_parser


if __name__=='__main__':
    # Set up the parameters for the experiment
    parser, _ = make_parser()
    conf = parser.parse_args(sys.argv[1:])
    device = conf.device
    timer = Timer()

    # Make necessary directories and files for logging
    os.makedirs(conf.save_dir, exist_ok=True) 
    orig_stdout = sys.stdout
    outfile = open(os.path.join(conf.save_dir, conf.log_file), 'w')
    sys.stdout=outfile

    # Make parameters that have to be calculated using other parameters
    conf.knowledge_vec_len = conf.M + 2*(conf.T-1)
    conf.noise_pwr_ff = 10**(-conf.snr_ff/10)
    conf.noise_pwr_fb = 10**(-conf.snr_fb/10)
    conf.test_batch_size = int(conf.batch_size / 10 * conf.K / conf.M)
    conf.num_training_samps = int(1000 * conf.batch_size)
    conf.num_iters_per_epoch = conf.num_training_samps // conf.batch_size

    # Make the TWC model
    model = Lightcode(conf).to(device)

    if conf.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        # Set up TensorBoard for logging purposes.
        writer = None
        if conf.use_tensorboard:
            log_folder = conf.save_dir + 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            writer = SummaryWriter()

    bs = conf.batch_size
    num_epochs = conf.num_epochs 
    grad_clip = conf.grad_clip 
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.optim_lr, weight_decay=conf.optim_weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: (1-epoch/conf.num_epochs))
    loss_fn = nn.CrossEntropyLoss()

    epoch_start = 0
    if conf.loadfile is not None:
        checkpoint = torch.load(conf.loadfile)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = checkpoint['epoch']

    bit_errors = []
    block_errors = []
    ctr = 0
    for epoch in range(epoch_start, num_epochs):
        model.train()
        losses = []
        for i in range(conf.num_iters_per_epoch):
            bitstreams_1 = torch.randint(0, 2, (bs, conf.M)).to(device)
            bitstreams_2 = torch.randint(0, 2, (bs, conf.M)).to(device)

            optimizer.zero_grad()
            output_1, output_2 = model(bitstreams_1, bitstreams_2)
            b_one_hot_1 = model.bits_to_one_hot(bitstreams_1).float()
            b_one_hot_2 = model.bits_to_one_hot(bitstreams_2).float()
            loss = loss_fn(output_1, b_one_hot_1) + loss_fn(output_2, b_one_hot_2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            losses.append(L:=loss.item()) # only works in Python >= 3.11 but easy to change
            optimizer.step()

            bit_estimates_1 = model.one_hot_to_bits(output_1).bool().view(bs,-1)
            bit_estimates_2 = model.one_hot_to_bits(output_2).bool().view(bs,-1)
            ber_1, bler_1 = model.calc_error_rates(bit_estimates_1, bitstreams_1.bool())
            ber_2, bler_2 = model.calc_error_rates(bit_estimates_2, bitstreams_2.bool())
            ber = ber_1 + ber_2
            bler = bler_1 + bler_2

            if i % 25 == 0:
                print(f'Epoch (iter): {epoch} ({i}), Loss: {L}, BER: {ber}, BLER: {bler}')

            if conf.use_tensorboard:
                writer.add_scalar('loss/train/BER', ber, ctr)
                writer.add_scalar('loss/train/BLER', bler, ctr)
                writer.add_scalar('loss/train/BER_1', ber_1, ctr)
                writer.add_scalar('loss/train/BLER_1', bler_1, ctr)
                writer.add_scalar('loss/train/BER_2', ber_2, ctr)
                writer.add_scalar('loss/train/BLER_2', bler_2, ctr)
                writer.add_scalar('loss/train/loss', L, ctr)
                ctr += 1
    
        ber_tup, bler_tup, _ = test_model(model=model, conf=conf)
        ber, ber_1, ber_2 = ber_tup
        bler, bler_1, bler_2 = bler_tup
        bit_errors.append(ber)
        block_errors.append(bler)

        scheduler.step()

        if conf.use_tensorboard:
            writer.add_scalar('loss/test/BER',ber,epoch)
            writer.add_scalar('loss/test/BLER',bler,epoch)
            writer.add_scalar('loss/test/BER_1',ber_1,epoch)
            writer.add_scalar('loss/test/BLER_1',bler_1,epoch)
            writer.add_scalar('loss/test/BER_2',ber_2,epoch)
            writer.add_scalar('loss/test/BLER_2',bler_2,epoch)

        print(f'\nEpoch Summary')
        print('====================================================')
        print(f'Epoch: {epoch}, Average loss: {np.mean(losses)}')
        print(f'BER: {ber:e}, BLER {bler:e}')
        print('====================================================\n')

        if epoch % conf.save_freq == 0:
            nowtime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            torch.save({'epoch' : epoch,
                        'model_state_dict' : model.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'scheduler_state_dict' : scheduler.state_dict(),
                        'loss' : L},
                        os.path.join(conf.save_dir, f'{nowtime}.pt'))
    
    nowtime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    torch.save({'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'loss' : L},
                os.path.join(conf.save_dir, f'{nowtime}.pt'))

    print(f'ber: {np.array(bit_errors)}')
    print(f'bler: {np.array(block_errors)}')
    b = {'ber' : np.array(bit_errors), 'bler' : np.array(block_errors)}
    with open(os.path.join(conf.save_dir, 'test_results.pkl'), 'wb') as f:
        pkl.dump(b,f)

    sys.stdout = orig_stdout
    outfile.close()