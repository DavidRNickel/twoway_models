import torch
import numpy as np

def test_model(model, conf):
    model.eval()
    ber = []
    ber_1 = []
    ber_2 = []
    bler = []
    bler_1 = []
    bler_2 = []
    xmit_pwr_1 = []
    xmit_pwr_2 = []
    
    if conf.is_one_way_active and conf.is_one_way_passive:
        raise AssertionError('Cannot have both active and passive feedback!')

    if not model.training:
        if conf.is_one_way_active: 
            conf.is_two_way = False
            print('Running one-way model with active feedback.')

        if conf.is_one_way_passive: 
            conf.is_two_way = False
            print('Running one-way model with passive feedback.')

        if conf.is_two_way == True:
            print('Runing two-way model.')

    with torch.no_grad():
        for e in range(conf.num_test_epochs):
            bits_1 = torch.randint(0, 2, (conf.test_batch_size, conf.K)).to(conf.device)
            if conf.is_two_way:
                bits_2 = torch.randint(0, 2, (conf.test_batch_size, conf.K)).to(conf.device)
            else: 
                bits_2 = torch.zeros(conf.test_batch_size, conf.K).to(conf.device)

            output_1, output_2 = model(2*bits_1.view(-1, conf.M)-1, 2*bits_2.view(-1, conf.M)-1)
            bit_estimates_1 = model.one_hot_to_bits(output_1).bool().view(-1, conf.K).detach().clone().cpu().numpy().astype(np.bool_)
            ber_tmp_1, bler_tmp_1 = model.calc_error_rates(bit_estimates_1, bits_1.detach().clone().cpu().numpy().astype(np.bool_))
            if conf.is_two_way:
                bit_estimates_2 = model.one_hot_to_bits(output_2).bool().view(-1, conf.K).detach().clone().cpu().numpy().astype(np.bool_)
                ber_tmp_2, bler_tmp_2 = model.calc_error_rates(bit_estimates_2, bits_2.detach().clone().cpu().numpy().astype(np.bool_))
            else:
                ber_tmp_2, bler_tmp_2 = 0,0

            ber_1.append(ber_tmp_1)
            ber_2.append(ber_tmp_2)
            bler_1.append(bler_tmp_1)
            bler_2.append(bler_tmp_2)
            ber.append(ber_tmp_1 + ber_tmp_2)
            bler.append(bler_tmp_1 + bler_tmp_2)
            xmit_pwr_1.append(model.xmit_pwr_track_1)
            xmit_pwr_2.append(model.xmit_pwr_track_2)
            
        ber = np.mean(ber)
        ber_1 = np.mean(ber_1)
        ber_2 = np.mean(ber_2)
        bler = np.mean(bler)
        bler_1 = np.mean(bler_1)
        bler_2 = np.mean(bler_2)
        xmit_pwr_1 = np.mean(xmit_pwr_1, axis=0)
        xmit_pwr_2 = np.mean(xmit_pwr_2, axis=0)
    
    return (ber, ber_1, ber_2), (bler, bler_1, bler_2), (xmit_pwr_1, xmit_pwr_2)
