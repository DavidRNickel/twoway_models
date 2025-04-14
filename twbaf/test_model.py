import torch
import numpy as np

def test_model(model, conf, show_progress_interval=None):
    model.eval()
    ber = []
    ber_1 = []
    ber_2 = []
    bler = []
    bler_1 = []
    bler_2 = []
    xmit_pwr_1 = []
    xmit_pwr_2 = []
    
    with torch.no_grad():
        for e in range(conf.num_test_epochs):
            bits_1 = torch.randint(0, 2, (conf.test_batch_size, conf.K)).to(conf.device)
            bits_2 = torch.randint(0, 2, (conf.test_batch_size, conf.K)).to(conf.device)
            b1 = bits_1.view(conf.test_batch_size, -1, conf.M)
            b2 = bits_2.view(conf.test_batch_size, -1, conf.M)
            output_1, output_2 = model(b1, b2)
            output_1 = output_1.view(-1, 2**model.M)
            output_2 = output_2.view(-1, 2**model.M)
            bit_estimates_1 = model.one_hot_to_bits(output_1).bool().view(-1, conf.K).detach().clone().cpu().numpy().astype(np.bool_)
            bit_estimates_2 = model.one_hot_to_bits(output_2).bool().view(-1, conf.K).detach().clone().cpu().numpy().astype(np.bool_)
            ber_tmp_1, bler_tmp_1 = model.calc_error_rates(bit_estimates_1, bits_1.detach().clone().cpu().numpy().astype(np.bool_))
            ber_tmp_2, bler_tmp_2 = model.calc_error_rates(bit_estimates_2, bits_2.detach().clone().cpu().numpy().astype(np.bool_))

            if show_progress_interval is not None:
                if e % show_progress_interval == 0:
                    print(int(e))

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
