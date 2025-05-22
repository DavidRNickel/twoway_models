import torch
import pickle as pkl
import sys, os, re

from gtwc_class import GTWC
from test_model import test_model
from calc_saved_mean_std import calc_mean_and_std
from make_argparser import make_parser

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
    outfile = open(os.path.join(conf.save_dir, conf.log_file), 'w')
    sys.stdout=outfile

    # Make parameters that have to be calculated using other parameters
    conf.knowledge_vec_len = conf.M + 2*(conf.T-1)
    conf.d_model = 32
    conf.noise_pwr_ff = 10**(-conf.snr_ff/10)
    conf.use_belief_network = False
    conf.noise_pwr_fb = 10**(-conf.snr_fb/10)
    conf.test_batch_size = conf.batch_size
    conf.num_training_samps = int(1000 * conf.batch_size)
    conf.num_iters_per_epoch = conf.num_training_samps // conf.batch_size
    conf.num_layers_xmit = 2 
    conf.num_layers_belief = 2
    conf.num_layers_recv = 3
    conf.n_heads = 1
    conf.d_model = 32
    conf.scaling_factor = 4
    conf.dropout = 0.0
    print(f'Num test epochs: {conf.num_test_epochs}')
    print(f'Batch size: {conf.test_batch_size}')

    model = GTWC(conf).to(conf.device)
    checkpoint = torch.load(conf.loadfile)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Getting saved mean and standard deviation...')
    ms = calc_mean_and_std(model, conf)
    model.mean_saved_1 = ms[0]
    model.std_saved_1 = ms[2]
    model.mean_saved_2 = ms[1]
    model.std_saved_2 = ms[3]
    print(f'Mean and STD: {ms}')

    print('Testing...')
    model.normalization_with_saved_data = True
    bers, blers, pwrs = test_model(model, conf, show_progress_interval=None)

    print(f'BER: {bers}')
    print(f'BLER: {blers}')
    print(f'Powers: {pwrs}')
    print(f'Sum Powers: {pwrs[0].sum()}, {pwrs[1].sum()}')
    # b = {'ber' : bers,
    #      'bler' : blers}
    # test_no = int(1)
    # with open(os.path.join(conf.save_dir, f'bler_{test_no}.pkl'), 'wb') as f:
    #     pkl.dump(b,f)

    sys.stdout = orig_stdout
    outfile.close()