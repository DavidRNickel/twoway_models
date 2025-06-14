import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from math import sqrt
import datetime
import sys, os, re
import pickle as pkl
from tqdm import tqdm

from gtwc_class import GTWC
from make_argparser import make_parser
from test_model import test_model


#
# cross-entropy loss with clipping to help prevent NAN
def custom_CE_loss(preds, targets):
    preds = F.softmax(preds, dim=1)
    return F.nll_loss(torch.log(preds.clip(1e-13, 1-1e-13)), targets.long()) # clip to avoid log(0)

if __name__=='__main__':
    parser, _ = make_parser()
    conf = parser.parse_args(sys.argv[1:])
    device = conf.device

    #
    # Make necessary directories and files for logging.
    # Doing this to avoid overwriting the existing training logs. 
    os.makedirs(conf.save_dir, exist_ok=True) 
    filenames = os.listdir(conf.save_dir)
    p = re.compile(f'^{conf.log_file}*')
    logfiles = sorted([s for s in filenames if p.match(s)])
    if len(logfiles) == 0:
        conf.log_file = conf.log_file + '.txt'
    elif len(logfiles) == 1:
        conf.log_file = conf.log_file.split('.')[0] + '_1.txt'
    else:
        lf = logfiles[-1].split('_')
        lf_name, lf_ext = '_'.join(lf[:-1]), lf[-1]# split apart the 
        n = int(lf_ext.split('.')[0])
        conf.log_file = lf_name + f'_{int(n+1)}.txt'
    orig_stdout = sys.stdout
    outfile = open(os.path.join(conf.save_dir, conf.log_file),'w')
    sys.stdout=outfile

    # Make parameters that have to be calculated using other parameters
    conf.knowledge_vec_len = conf.M + 2*(conf.T-1)
    conf.d_model = 32
    conf.noise_pwr_ff = 10**(-conf.snr_ff/10)
    conf.use_belief_network = False
    conf.noise_pwr_fb = 10**(-conf.snr_fb/10)
    conf.test_batch_size = conf.batch_size
    conf.num_iters_per_epoch = 1000
    conf.num_layers_xmit = 2 
    conf.num_layers_belief = 2
    conf.num_layers_recv = 3
    conf.n_heads = 1
    conf.d_model = 32
    conf.scaling_factor = 4
    conf.dropout = 0.0

    
    gtwc = GTWC(conf).to(device)
    if 'cuda' in device: # just in case the device is something like cuda:0
        torch.backends.cudnn.benchmark = True
    nowtime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    if conf.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        # Set up TensorBoard for logging purposes.
        writer = None
        if conf.use_tensorboard:
            log_folder = 'logs/fit/' + nowtime
            writer = SummaryWriter()

    bs = conf.batch_size
    num_epochs = conf.num_epochs
    grad_clip = conf.grad_clip

    num_epochs = conf.num_epochs 
    optimizer = torch.optim.AdamW(gtwc.parameters(), lr=conf.optim_lr, weight_decay=conf.optim_weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: (1-epoch/conf.num_epochs))
    loss_fn = nn.CrossEntropyLoss()

    epoch_start = 0
    if conf.loadfile is not None:
        checkpoint = torch.load(conf.loadfile)
        gtwc.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = checkpoint['epoch']

    bit_errors = []
    block_errors = []
    ctr = 0
    map_vec = 2**(torch.arange(conf.M)).flip(0).float().to(device)
    pbar = tqdm(total = num_epochs-epoch_start)
    for epoch in range(epoch_start, num_epochs):
        pbar.update(1)
        gtwc.train()
        losses = []
        # Since bs >> 2^M, we will see sufficiently many examples of each.
        # message. Thebits_to_one_hot conversion takes orders of magnitude more 
        # time to execute than everything else, so I had to move it out here.
        bitstreams_1 = torch.randint(0, 2, (bs, conf.K), device=device)
        bitstreams_2 = torch.randint(0, 2, (bs, conf.K), device=device)
        b1, b2 = bitstreams_1.view(bs,-1,conf.M), bitstreams_2.view(bs,-1,conf.M)
        b_target_1 = b1.view(-1, conf.M).float() @ map_vec
        b_target_2 = b2.view(-1, conf.M).float() @ map_vec
        for i in range(conf.num_iters_per_epoch):
            optimizer.zero_grad()
            output_1, output_2 = gtwc(b1, b2)
            output_1 = output_1.view(bs*gtwc.num_blocks, 2**gtwc.M)
            output_2 = output_2.view(bs*gtwc.num_blocks, 2**gtwc.M)
            # loss = loss_fn(output_1, b_one_hot_1) + loss_fn(output_2, b_one_hot_2)
            loss = custom_CE_loss(output_1, b_target_1) + custom_CE_loss(output_2, b_target_2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(gtwc.parameters(), grad_clip)
            losses.append(L:=loss.item()) # only works in Python >= 3.11
            optimizer.step()

            if i % 100 == 0:
                bit_estimates_1 = gtwc.one_hot_to_bits(output_1).bool().view(bs,-1)
                bit_estimates_2 = gtwc.one_hot_to_bits(output_2).bool().view(bs,-1)
                ber_1, bler_1 = gtwc.calc_error_rates(bit_estimates_1, bitstreams_1.bool())
                ber_2, bler_2 = gtwc.calc_error_rates(bit_estimates_2, bitstreams_2.bool())
                ber = ber_1 + ber_2
                bler = bler_1 + bler_2

                if conf.use_tensorboard:
                    writer.add_scalar('loss/train/BER', ber, ctr)
                    writer.add_scalar('loss/train/BLER', bler, ctr)
                    writer.add_scalar('loss/train/BER_1', ber_1, ctr)
                    writer.add_scalar('loss/train/BLER_1', bler_1, ctr)
                    writer.add_scalar('loss/train/BER_2', ber_2, ctr)
                    writer.add_scalar('loss/train/BLER_2', bler_2, ctr)
                    writer.add_scalar('loss/train/loss', L, ctr)
                    ctr += 1

                print(f'Epoch (iter): {epoch} ({i}), Loss: {L}, BER: {ber}, BLER: {bler}')
    
        ber_tup, bler_tup, _ = test_model(model=gtwc, conf=conf)
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
        print('====================================================\n'); nowtime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        if epoch % conf.save_freq == 0:
            nowtime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            torch.save({'epoch' : epoch+1,
                        'model_state_dict' : gtwc.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'scheduler_state_dict' : scheduler.state_dict(),
                        'loss' : L},
                        os.path.join(conf.save_dir, f'{nowtime}.pt'))
        
    
    nowtime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    torch.save({'epoch' : epoch+1,
                'model_state_dict' : gtwc.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'loss' : L},
                os.path.join(conf.save_dir, 'final.pt'))

    print(f'ber: {np.array(bit_errors)}')
    print(f'bler: {np.array(block_errors)}')
    b = {'ber' : np.array(bit_errors), 'bler' : np.array(block_errors)}
    # with open(os.path.join(conf.save_dir, 'test_results.pkl'), 'wb') as f:
    #     pkl.dump(b,f)

    sys.stdout = orig_stdout
    outfile.close()
