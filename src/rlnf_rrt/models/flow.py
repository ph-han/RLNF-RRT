import torch
import torch.nn as nn

from rlnf_rrt.models.cond_encoder import CondEncoder
from rlnf_rrt.models.affine_coupling_block import AffineCouplingBlock



class Flow(nn.Module):
    def __init__(
        self,
        num_blocks: int = 4,
        latent_dim: int = 128,
        sg_dim: int = 2,
        hidden_dim: int = 128,
        s_max: float = 1.5,
        channels=(32, 48, 64, 96, 128),
        norm: str = "bn",
    ):
        super().__init__()

        self.cond_encoder = CondEncoder(
            sg_dim=sg_dim,
            latent_dim=latent_dim,
            channels=channels,
            norm=norm,
        )
        self.cond_dim = latent_dim + 2 * sg_dim
        self.flows = nn.ModuleList([
            AffineCouplingBlock(
                cond_dim=self.cond_dim,
                hidden_dim=hidden_dim,
                s_max=s_max
            )
            for _ in range(num_blocks)
        ])

    def forward(self, binary_map: torch.Tensor, start: torch.Tensor, goal: torch.Tensor, x: torch.Tensor):
        cond = self.cond_encoder(binary_map, start, goal)
        
        tot_log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        for flow in self.flows:
            z, log_det = flow(z, cond)
            tot_log_det += log_det

        return z, tot_log_det
    
    @torch.no_grad()
    def inverse(self, map_img: torch.Tensor, start: torch.Tensor, goal: torch.Tensor, z: torch.Tensor):
        cond = self.cond_encoder(map_img, start, goal)

        tot_log_det = z.new_zeros(z.shape[0])
        x = z
        for block in reversed(self.flows):
            x, log_det = block.inverse(x, cond)
            tot_log_det += log_det
        return x, tot_log_det

    
        