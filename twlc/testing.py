import torch
import pickle as pkl
import sys, os, re

from lightcode_class import Lightcode
from test_model import test_model
from calc_saved_mean_std import calc_mean_and_std
from make_argparser import make_parser

if __name__=='__main__':
    parser, _ = make_parser()
    conf = parser.parse_args(sys.argv[1:])
    device = conf.device

    if conf.is_one_way_active and conf.is_one_way_passive:
        raise AssertionError('Cannot have both active and passive feedback!')

    if conf.is_one_way_active: 
        conf.is_two_way, conf.is_one_way_passive = False, False
        print('Running one-way model with active feedback. ',end='')

    elif conf.is_one_way_passive: 
        conf.is_two_way, conf.is_one_way_active = False, False
        print('Running one-way model with passive feedback. ',end='')

    else:
        conf.is_one_way_passive, conf.is_one_way_active = False, False
        print('Runing two-way model. ',end='')

    print(f'K: {conf.K}, M: {conf.M}, T: {conf.T}, SNR1: {conf.snr_ff}, SNR2: {conf.snr_fb}')

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

    # Make parameters that have to be calculated using other parameters
    conf.knowledge_vec_len = conf.M + 2*(conf.T-1)
    conf.noise_pwr_ff = 10**(-conf.snr_ff/10)
    conf.noise_pwr_fb = 10**(-conf.snr_fb/10)
    conf.test_batch_size = conf.batch_size 
    print(f'Num test epochs: {conf.num_test_epochs}')
    print(f'Batch size: {conf.batch_size}')

    model = Lightcode(conf).to(device)
    checkpoint = torch.load(conf.loadfile)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Getting saved mean and standard deviation...')
    ms = calc_mean_and_std(model, conf)
    model.mean_saved_1 = ms[0]
    model.std_saved_1 = ms[2]
    if not conf.is_one_way_passive:
        model.mean_saved_2 = ms[1]
        model.std_saved_2 = ms[3]
    else:
        model.mean_saved_2 = None 
        model.std_saved_2 = None
    print(f'Mean and STD: {ms}')

    print('Testing...')
    model.normalization_with_saved_data = True
    bers, blers, pwrs = test_model(model, conf)

    print(f'BER: {bers}')
    print(f'BLER: {blers}')
    print(f'Powers: {pwrs}')
    print(f'Sum Powers: {pwrs[0].sum()}, {pwrs[1].sum()}')

    sys.stdout = orig_stdout
    outfile.close()