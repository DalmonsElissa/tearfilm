"""Divided space-time video transformer.

Ref: Table 2, row "Video Transformer" [49], family *full attention*.

Attention is divided: each token attends across time at a fixed spatial
position, then across space within its frame. That is the design the comparison
binds to its published form, and its clip-level cost is why the table's baseline
budget is several times the constrained one.
"""

from __future__ import annotations

import torch
from torch import nn

from tearfilm_persistence.comparators.tower import ClipHead


class DividedAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.time_norm = nn.LayerNorm(dim)
        self.time_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.space_norm = nn.LayerNorm(dim)
        self.space_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        steps: int,
        spatial: int,
        frame_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, dim = tokens.shape
        temporal = tokens.reshape(batch, steps, spatial, dim).permute(0, 2, 1, 3)
        temporal = temporal.reshape(batch * spatial, steps, dim)
        key_mask = (~frame_valid).unsqueeze(1).expand(batch, spatial, steps)
        key_mask = key_mask.reshape(batch * spatial, steps)
        attended, _ = self.time_attention(
            self.time_norm(temporal),
            self.time_norm(temporal),
            self.time_norm(temporal),
            key_padding_mask=key_mask,
            need_weights=False,
        )
        temporal = temporal + attended
        tokens = temporal.reshape(batch, spatial, steps, dim).permute(0, 2, 1, 3)
        tokens = tokens.reshape(batch * steps, spatial, dim)
        attended, _ = self.space_attention(
            self.space_norm(tokens),
            self.space_norm(tokens),
            self.space_norm(tokens),
            need_weights=False,
        )
        tokens = (tokens + attended).reshape(batch, steps, spatial, dim)
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        return tokens.reshape(batch, steps * spatial, dim)


class VideoTransformer(nn.Module):
    family = "full attention"

    def __init__(
        self,
        dim: int = 48,
        depth: int = 2,
        heads: int = 4,
        patch: int = 16,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.embed = nn.Conv3d(3, dim, (1, patch, patch), stride=(1, patch, patch))
        self.blocks = nn.ModuleList([DividedAttentionBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = ClipHead(dim)
        self.dim = dim

    def forward(
        self, frames: torch.Tensor, frame_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps = frames.shape[0], frames.shape[1]
        clip = frames.transpose(1, 2)
        embedded = self.embed(clip)
        grid_h, grid_w = embedded.shape[-2], embedded.shape[-1]
        spatial = grid_h * grid_w
        tokens = embedded.flatten(2).transpose(1, 2).reshape(batch, steps, spatial, self.dim)
        tokens = tokens.reshape(batch, steps * spatial, self.dim)
        for block in self.blocks:
            tokens = block(tokens, steps, spatial, frame_mask)
        tokens = self.norm(tokens).reshape(batch, steps, spatial, self.dim)
        weights = frame_mask.to(tokens.dtype).unsqueeze(-1).unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = pooled.mean(dim=1)
        logit, onset = self.head(pooled)
        return logit, onset
