import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .RealNVP import RealNVP

class ConditionalNF(nn.Module):
    def __init__(self, masks, hidden_dim):
        super().__init__()