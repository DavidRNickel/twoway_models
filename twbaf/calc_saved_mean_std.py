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
            bits_1 = torch.randint(0, 2, (conf.test_batch_size, conf.K), device=conf.device)
            bits_2 = torch.randint(0, 2, (conf.test_batch_size, conf.K), device=conf.device)
            b1 = bits_1.view(conf.test_batch_size, -1, conf.M)
            b2 = bits_2.view(conf.test_batch_size, -1, conf.M)
            _, _= model(b1, b2)
            
            mean_tot_1 += model.mean_batch_1
            mean_tot_2 += model.mean_batch_2
            std_tot_1 += model.std_batch_1
            std_tot_2 += model.std_batch_2
    
    nve = conf.num_valid_epochs
    mean_1 = mean_tot_1 / nve
    mean_2 = mean_tot_2 / nve
    std_1 = std_tot_1 / nve
    std_2 = std_tot_2 / nve

    return mean_1, mean_2, std_1, std_2