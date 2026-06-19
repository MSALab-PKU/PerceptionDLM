import re
import torch
from torch import nn
from torch.nn import functional as F


def build_projection(projection_type: str, in_dim: int, out_dim: int) -> nn.Module:
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projection_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(in_dim, out_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(out_dim, out_dim))
        projection = nn.Sequential(*modules)
        return projection

    raise ValueError(f'Unknown projector type: {projection_type}')


class PerceiverProjection(nn.Module):
    def __init__(self, projection_type: str, in_dim: int, out_dim: int):
        super().__init__()
        self.projection = build_projection(projection_type, in_dim, out_dim)

    def forward(self, input_embeds: torch.Tensor):
        input_embeds.requires_grad_(True)
        embeds = self.projection(input_embeds)
        embeds.requires_grad_(True)
        return embeds