import torch
import torch.nn as nn

class ConditionalAffineCouplingLayer(nn.Module):
    def __init__(self, mask, hidden_dim, condition_dim, clamp=2.0):
        super().__init__()

        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.clamp = clamp

        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32))

        self.condition_input_dim = self.input_dim + self.condition_dim

        self.injection_condi = nn.Sequential(
            nn.Linear(self.condition_input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.s_net = nn.Linear(self.hidden_dim, self.input_dim)
        self.t_net = nn.Linear(self.hidden_dim, self.input_dim)


        self.s_scale = nn.Parameter(torch.zeros(self.input_dim))
        nn.init.normal_(self.s_scale, mean=0.0, std=0.01)

    def _f_clamp(self, s):
        return self.clamp * (0.6366 * torch.atan(s))

    def forward(self, x, condition):
        x_masked = x * self.mask
        condition_input = self.injection_condi(torch.cat([x_masked, condition], dim=-1))

        s = self._f_clamp(self.s_net(condition_input)) * self.s_scale
        t = self.t_net(condition_input)

        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det_jacob = torch.sum((1 - self.mask) * s, dim=-1)
        return y, log_det_jacob
    
    def inverse(self, y, condition):
        y_masked = y * self.mask
        condition_input = self.injection_condi(torch.cat([y_masked, condition], dim=-1))

        s = self._f_clamp(self.s_net(condition_input)) * self.s_scale
        t = self.t_net(condition_input)

        x = y_masked + (1 - self.mask) * (y - t) * torch.exp(-s)
        log_det_jacob = -torch.sum((1 - self.mask) * s, dim=-1)
        return x, log_det_jacob
