import torch
import numpy as np

def calc_mean_and_std(model, conf):
    mean_tot_1 = 0
    mean_tot_2 = 0
    std_tot_1 = 0
    std_tot_2 = 0

    model.eval()
    with torch.no_grad():
        for _ in range(conf.num_valid_epochs):
            bits_1 = torch.randint(0, 2, (conf.test_batch_size, conf.K)).to(conf.device)
            if conf.is_two_way:
                bits_2 = torch.randint(0, 2, (conf.test_batch_size, conf.K)).to(conf.device)
            else:
                bits_2 = torch.zeros(conf.test_batch_size, conf.K).to(conf.device)
            b1 = bits_1.view(-1, conf.M)
            b2 = bits_2.view(-1, conf.M)
            _, _= model(2*b1-1, 2*b2-1)
            
            mean_tot_1 += model.mean_batch_1
            std_tot_1 += model.std_batch_1
            if not model.is_one_way_passive:
                std_tot_2 += model.std_batch_2
                mean_tot_2 += model.mean_batch_2
    
    nve = conf.num_valid_epochs
    mean_1 = mean_tot_1 / nve
    std_1 = std_tot_1 / nve
    if not model.is_one_way_passive:
        mean_2 = mean_tot_2 / nve
        std_2 = std_tot_2 / nve
    
    else: std_2, mean_2 = 0,0

    return mean_1, mean_2, std_1, std_2