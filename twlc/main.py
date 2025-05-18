import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import sys, os, re, copy
import pickle as pkl
from datetime import datetime
from tqdm import tqdm

from lightcode_class import Lightcode
from test_model import test_model
from make_argparser import make_parser


#
# To do one-way with active feedback, I just run the two-way model but don't
# add in the loss wrt decoder 2 into the objective. I'm sure there are ways to
# make the code faster, but this kept it as readable as possible.
#

# print to stderr during training
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# cross-entropy loss with clipping to help prevent NAN
def custom_CE_loss(preds, targets):
    preds = F.softmax(preds, dim=1)
    return F.nll_loss(torch.log(preds.clip(1e-13, 1-1e-13)), targets.long()) # clip to avoid log(0)

if __name__=='__main__':
    # Set up the parameters for the experiment
    parser, _ = make_parser()
    conf = parser.parse_args(sys.argv[1:])
    device = conf.device

    if conf.is_one_way_active and conf.is_one_way_passive:
        raise AssertionError('Cannot have both active and passive feedback!')

    if conf.is_one_way_active: 
        conf.is_two_way, conf.is_one_way_passive = False, False
        print('Running one-way model with active feedback.')

    elif conf.is_one_way_passive: 
        conf.is_two_way, conf.is_one_way_active = False, False
        print('Running one-way model with passive feedback.')

    else:
        conf.is_one_way_passive, conf.is_one_way_active = False, False
        print('Runing two-way model.')

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

    # Now redirect the printout to the correct location.
    # Probably should have done this with logging utilities.
    orig_stdout = sys.stdout
    outfile = open(os.path.join(conf.save_dir, conf.log_file), 'w')
    sys.stdout=outfile

    # Print out system information
    print(f'Running on: {torch.cuda.get_device_name()}\n')

    # Make parameters that have to be calculated using other parameters
    conf.knowledge_vec_len = conf.M + 2*(conf.T-1)
    conf.noise_pwr_ff = 10**(-conf.snr_ff/10) 
    conf.noise_pwr_fb = 10**(-conf.snr_fb/10)
    conf.test_batch_size = 10
    print(f'Num training epochs: {conf.num_epochs}')
    print(f'Num iters per epoch: {conf.num_iters_per_epoch}')
    print(f'Batch size: {conf.batch_size}')

    # Make the TWC model
    model = Lightcode(conf).to(device)
    if device=='cuda':
        torch.backends.cudnn.benchmark = True

    # Set up tensorboard if you want to use it during training. 
    if conf.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        # Set up TensorBoard for logging purposes.
        writer = SummaryWriter(log_dir = os.path.join(conf.save_dir, f"tblog_{datetime.now().strftime('%Y%m%d-%H%M%S')}"))

    # Training parameters.
    bs = conf.batch_size
    num_epochs = conf.num_epochs 
    grad_clip = conf.grad_clip 
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.optim_lr, weight_decay=conf.optim_weight_decay, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: (1-epoch/(conf.num_epochs*conf.num_iters_per_epoch)))
    print(f'Optimizer (decay): {optimizer} ({conf.optim_weight_decay})')
    print(f'Scheduler: {scheduler}')
    print(f'Learning-rate: {conf.optim_lr}')
    print(f'Gradient clip: {conf.grad_clip}')

    # Load in previously saved model if training got halted at some point.
    epoch_start = 0
    ctr = 0 # used for indexing tensorboard writer
    if conf.loadfile is not None:
        checkpoint = torch.load(conf.loadfile)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = checkpoint['epoch']
        ctr = epoch_start * conf.num_iters_per_epoch

    map_vec = 2**(torch.arange(conf.M)).flip(0).float().to(device)
    pbar = tqdm(total = num_epochs-epoch_start)
    total_run_starttime = datetime.now()
    
    for epoch in range(epoch_start, num_epochs):
        pbar.update(1)
        model.train()
        for i in range(conf.num_iters_per_epoch):
            
            # Make the bitstreams
            bitstreams_1 = torch.randint(0, 2, (bs, conf.K), device=device)
            b_target_1 = bitstreams_1.view(-1, conf.M).float() @ map_vec
            if conf.is_two_way:
                bitstreams_2 = torch.randint(0, 2, (bs, conf.K), device=device)
                b_target_2 = bitstreams_2.view(-1, conf.M).float() @ map_vec
            else:
                # Not used in one_way_passive. bitstreams_2 is used as an input
                # for the feedback encoder to keep the code concise; the 
                # corresponding weights atrophy and don't impact overall performance.
                bitstreams_2 = torch.zeros((bs, conf.K), device=device)
                b_target_2 = bitstreams_2.view(-1, conf.M).float() @ map_vec

            optimizer.zero_grad()

            output_1, output_2 = model(2*bitstreams_1.view(-1, conf.M)-1, 
                                       2*bitstreams_2.view(-1, conf.M)-1)

            loss = custom_CE_loss(output_1, b_target_1.long()) 
            if conf.is_two_way:
                loss += custom_CE_loss(output_2, b_target_2.long())

            if np.isnan(loss.item()):
                print('Encountered NAN...')
                eprint('Encountered NAN...')
                sys.exit()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            if i % 100 == 0:
                with torch.no_grad():
                    bit_estimates_1 = model.one_hot_to_bits(output_1).bool().view(bs,-1)
                    ber_1, bler_1 = model.calc_error_rates(bit_estimates_1, bitstreams_1.bool())
                    if conf.is_two_way:
                        bit_estimates_2 = model.one_hot_to_bits(output_2).bool().view(bs,-1)
                        ber_2, bler_2 = model.calc_error_rates(bit_estimates_2, bitstreams_2.bool())
                    else:
                        ber_2, bler_2 = 0, 0
                    ber = ber_1 + ber_2
                    bler = bler_1 + bler_2
                    print(f'Epoch (iter): {epoch} ({i}), Loss: {loss.item()}, BER: {ber}, BLER: {bler}')
                    # eprint(f'Epoch (iter): {epoch} ({i}), Loss: {loss.item()}, BER: {ber}, BLER: {bler}')

                    if conf.use_tensorboard:
                        writer.add_scalar('loss/train/BER', ber, ctr)
                        writer.add_scalar('loss/train/BLER', bler, ctr)
                        writer.add_scalar('loss/train/BER_1', ber_1, ctr)
                        writer.add_scalar('loss/train/BLER_1', bler_1, ctr)
                        writer.add_scalar('loss/train/BER_2', ber_2, ctr)
                        writer.add_scalar('loss/train/BLER_2', bler_2, ctr)

            if conf.use_tensorboard:
                writer.add_scalar('loss/train/loss', loss.item(), ctr)

            ctr += 1
    
        ber_tup, bler_tup, _ = test_model(model=model, conf=conf)
        ber, ber_1, ber_2 = ber_tup
        bler, bler_1, bler_2 = bler_tup


        if conf.use_tensorboard:
            writer.add_scalar('loss/test/BER',ber,epoch)
            writer.add_scalar('loss/test/BLER',bler,epoch)
            writer.add_scalar('loss/test/BER_1',ber_1,epoch)
            writer.add_scalar('loss/test/BLER_1',bler_1,epoch)
            writer.add_scalar('loss/test/BER_2',ber_2,epoch)
            writer.add_scalar('loss/test/BLER_2',bler_2,epoch)

        # eprint('')
        print(f'\nEpoch Summary')
        print('====================================================')
        print(f'Epoch: {epoch}, Average loss: {loss.item()}')
        print(f'BER: {ber:e}, BLER {bler:e}')
        print('====================================================\n')

        if epoch % conf.save_freq == 0:
            nowtime = datetime.now().strftime('%Y%m%d-%H%M%S')
            torch.save({'epoch' : epoch+1,
                        'model_state_dict' : model.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'scheduler_state_dict' : scheduler.state_dict(),
                        'loss' : loss.item()},
                        os.path.join(conf.save_dir, f'{nowtime}.pt'))
    
    nowtime = datetime.now().strftime('%Y%m%d-%H%M%S')
    torch.save({'epoch' : epoch+1,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'loss' : loss.item()},
                os.path.join(conf.save_dir, 'final.pt'))
    
    print(f'\nTotal runtime: {datetime.now()-total_run_starttime}')

    sys.stdout = orig_stdout
    outfile.close()
