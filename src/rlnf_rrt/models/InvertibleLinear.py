import torch
import torch.nn as nn

class InvertibleLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # orthogonal init → det = ±1 → 안정적
        W = torch.linalg.qr(torch.randn(dim, dim))[0]
        self.W = nn.Parameter(W)

    def forward(self, x):
        # x: (B, D)
        y = x @ self.W
        log_det = torch.slogdet(self.W)[1]
        return y, log_det.expand(x.size(0))

    def inverse(self, y):
        W_inv = torch.inverse(self.W)
        x = y @ W_inv
        log_det = -torch.slogdet(self.W)[1]
        return x, log_det.expand(y.size(0))
