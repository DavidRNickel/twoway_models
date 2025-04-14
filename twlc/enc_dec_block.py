import torch
from torch import nn
from torch.nn import functional as F

class EncDecBlock(nn.Module):
    def __init__(self, input_dim, output_dim, conf, is_encoder=False, d_model=16):
        super(EncDecBlock, self).__init__()

        self.relu = nn.ReLU()
        self.is_encoder = is_encoder
        self.M = conf.M

        # Implementation per Fig. 14 of ArXiV paper
        self.fe_fc1 = nn.Linear(input_dim, 2*d_model)
        self.fe_fc2 = nn.Linear(2*d_model, 2*d_model)
        self.fe_fc3 = nn.Linear(2*d_model, 2*d_model)
        self.fe_fc4 = nn.Linear(4*d_model, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        
        if not self.is_encoder:
            self.mlp_fc1 = nn.Linear(d_model, d_model)
        self.mlp_out = nn.Linear(d_model, output_dim)


    def forward(self, src):
        x1 = self.fe_fc1(src)
        src1 = -1 * x1
        x = self.fe_fc2(self.relu(x1))
        x = self.fe_fc3(self.relu(x))
        x = self.fe_fc4(torch.hstack((x,src1)))
        x = self.norm(x)

        # MLP
        if not self.is_encoder:
            x = self.mlp_fc1(self.relu(x))
        x = self.mlp_out(self.relu(x))

        return x