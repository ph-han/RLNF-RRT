import torch
import torch.nn as nn

class AffineCouplingLayer(nn.Module):
    def __init__(self, mask, hidden_dim):
        super().__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim

        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32))

        self.s_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Tanh()
        )

        self.t_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim)
        )
        self.s_scale = nn.Parameter(torch.zeros(self.input_dim))
        nn.init.normal_(self.s_scale, mean=0.0, std=0.01)

    def forward(self, x):
        x_masked = x * self.mask

        s = self.s_net(x_masked) * self.s_scale
        t = self.t_net(x_masked)

        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det_jacob = torch.sum((1 - self.mask) * s, dim=1)
        return y, log_det_jacob
    
    def inverse(self, y):
        y_masked = y * self.mask

        s = self.s_net(y_masked) * self.s_scale
        t = self.t_net(y_masked)

        x = y_masked + (1 - self.mask) * (y - t) * torch.exp(-s)
        log_det_jacob = -torch.sum((1 - self.mask) * s, dim=1)
        return x, log_det_jacob

class RealNVP(nn.Module):
    def __init__(self, masks, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            AffineCouplingLayer(mask, self.hidden_dim) for mask in masks
        ])

    def forward(self, x):
        y = x

        log_det_tot = 0
        for layer in self.layers:
            y, log_det_jacob = layer(y)
            log_det_tot += log_det_jacob


        log_det_tanh = torch.sum(torch.log(torch.abs(4 * (1 - torch.tanh(y)**2))), dim=-1)
        y = 4 * torch.tanh(y)
        log_det_tot += log_det_tanh
        
        return y, log_det_tot
    
    def inverse(self, y):
        x = y

        log_det_tot = 0
        log_det_tanh_inv = torch.sum(torch.log(torch.abs(0.25 * 1 / (1 - (x/4)**2))), dim=-1)
        x = 0.5 * torch.log((1 + x/4) / (1 - x/4))
        log_det_tot += log_det_tanh_inv

        for layer in reversed(self.layers):
            x, log_det_jacob = layer.inverse(x)
            log_det_tot += log_det_jacob

        return x, log_det_tot
