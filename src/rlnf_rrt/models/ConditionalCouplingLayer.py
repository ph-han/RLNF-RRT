import torch
import torch.nn as nn

class ConditionalAffineCouplingLayer(nn.Module):
    def __init__(self, mask, hidden_dim, condition_dim, s_max=2.0):
        super().__init__()

        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.s_max = s_max

        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32))

        self.condition_input_dim = self.input_dim + self.condition_dim
        self.s_net = nn.Sequential(
            nn.Linear(self.condition_input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(self.hidden_dim, self.input_dim)
        )

        self.t_net = nn.Sequential(
            nn.Linear(self.condition_input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(self.hidden_dim, self.input_dim)
        )


        self.s_gain = nn.Parameter(torch.ones(self.input_dim))


    def forward(self, x, condition):
        x_masked = x * self.mask

        coupling_input = torch.cat([x_masked, condition], dim=-1)

        s = torch.tanh(self.s_net(coupling_input)) * self.s_max * self.s_gain
        t = self.t_net(coupling_input)

        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det_jacob = torch.sum((1 - self.mask) * s, dim=-1)
        return y, log_det_jacob
    
    def inverse(self, y, condition):
        y_masked = y * self.mask
        
        coupling_input = torch.cat([y_masked, condition], dim=-1)

        s = torch.tanh(self.s_net(coupling_input)) * self.s_max * self.s_gain
        t = self.t_net(coupling_input)

        x = y_masked + (1 - self.mask) * (y - t) * torch.exp(-s)
        log_det_jacob = -torch.sum((1 - self.mask) * s, dim=-1)
        return x, log_det_jacob
