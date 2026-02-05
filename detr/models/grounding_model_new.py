import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def make_xy_pos_grid(h: int, w: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Returns (1, 2, h, w) grid with x,y in [-1, 1].
    channel 0: x, channel 1: y
    """
    ys = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=0).unsqueeze(0)  # (1,2,h,w)
    return grid

class GroundingModel(nn.Module):
    """
    Implements:
    img -> f(B,5,8,20)
    m = sigmoid(obj_head(f)) (B,1,8,20)
    f_g = f * m
    concat pos (B,2,8,20) -> h(B,7,8,20)
    proj -> u(B,d_model,8,20)
    flatten -> z0(B,160,d_model)
    self-attn -> z(B,160,d_model)

    text tokens: (color_id, left_id, right_id) -> embeddings -> t(B,3,d_model)
    per-token scoring vs z -> S_k(B,1,8,20), k=0..2
    weighted sum with learnable weights -> final logits heatmap(B,1,8,20)
    """
    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        score_dim: int = 32,  # inner dim for dot-product scoring
        n_colors: int = 3,    # e.g., {pad, orange, blue}
        n_left_ord: int = 6,  # e.g., {pad, 1st, 2nd, 3rd, ...}
        n_right_ord: int = 6, # e.g., {pad, 1st, 2nd, 3rd, ...}
        grid_hw: Tuple[int, int] = (8, 20),
    ):
        super().__init__()
        self.H, self.W = grid_hw
        self.N = self.H * self.W
        self.d_model = d_model
        self.score_dim = score_dim

        # 4) self-attention over 160 tokens
        # batch_first=True => input/output (B, N, d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.self_attn_ln = nn.LayerNorm(d_model)
        self.self_attn_ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
        )
        self.self_attn_ff_ln = nn.LayerNorm(d_model)

        # 5) text embeddings (3 independent tokens, no pooling/self-attn)
        self.emb_color = nn.Embedding(n_colors, d_model)
        self.emb_left = nn.Embedding(n_left_ord, d_model)
        self.emb_right = nn.Embedding(n_right_ord, d_model)

        # 6) scoring projections
        # Keep tokens independent by giving each token its own Q projection.
        self.q_proj = nn.ModuleList([nn.Linear(d_model, score_dim) for _ in range(3)])
        self.k_proj = nn.Linear(d_model, score_dim)

        # 7) learnable weights for combining 3 maps (stable: use softmax)
        self.map_weights = nn.Parameter(torch.zeros(3))  # start equal after softmax

        # register pos grid as buffer (created lazily in forward if device changes)
        self.register_buffer("_pos_grid", torch.empty(0), persistent=False)

        self.cross_attn_ln = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.feat_proj = nn.Conv2d(512, d_model, kernel_size=1)
        self.out_proj = nn.Conv2d(513, 512, kernel_size=1)
        self.grid_proj = nn.Conv2d(2, 512, kernel_size=1)

    def _get_pos_grid(self, device, dtype) -> torch.Tensor:
        if self._pos_grid.numel() == 0 or self._pos_grid.device != device or self._pos_grid.dtype != dtype:
            self._pos_grid = make_xy_pos_grid(self.H, self.W, device=device, dtype=dtype)  # (1,2,H,W)
        return self._pos_grid
    
    def forward(
        self,
        feat: torch.Tensor,          # (B,512,8,20)
        pos: torch.Tensor,           # (B,1,8,20) or (1,1,8,20)
        color_id: torch.Tensor,      # (B,)
        left_ord_id: torch.Tensor,   # (B,)
        right_ord_id: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        start_time = time.time()
        B, C, H, W = feat.shape
        assert (H, W) == (8, 20)
        # feat = self.feat_proj(feat)
        # 1) add positional encoding
        x = feat + pos               # (B,512,8,20)

        pos_grid = self._get_pos_grid(device=feat.device, dtype=feat.dtype)  # (1,2,H,W)
        grid_emb = self.grid_proj(pos_grid)  # (1,d_model,H,W), broadcastable to B
        x = feat + pos + grid_emb

        # tokenize
        z0 = x.flatten(2).transpose(1, 2)  # (B,N,512), N=H*W

        # 2) self-attention over spatial tokens
        z1 = self.self_attn_ln(z0)
        sa_out, _ = self.self_attn(z1, z1, z1, need_weights=False)
        z = z0 + sa_out

        z2 = self.self_attn_ff_ln(z)
        z = z + self.self_attn_ff(z2)       # (B,N,512)

        # 3) text / id tokens
        t_color = self.emb_color(color_id)       # (B,512)
        t_left  = self.emb_left(left_ord_id)     # (B,512)
        t_right = self.emb_right(right_ord_id)   # (B,512)

        t = torch.stack([t_color, t_left, t_right], dim=1)  # (B,3,512)

        S_list = []
        for i, q_proj in enumerate(self.q_proj):
            Q = q_proj(t[:, i, :])            # (B, score_dim)
            K = self.k_proj(z)                 # (B, N, score_dim)
            # scaled dot-product: B x N
            S = torch.einsum('bd,bnd->bn', Q, K) / math.sqrt(self.score_dim)
            S_list.append(S.unsqueeze(1))      # (B,1,N)

        # stack into (B, 3, N)
        S = torch.cat(S_list, dim=1)

        weights = torch.softmax(self.map_weights, dim=0)  # sum to 1
        S_weighted = (S * weights[None, :, None]).sum(dim=1)  # (B, N) 
        S_map = S_weighted.view(B, 1, H, W)  # (B,1,H,W)

        S_map_flat = S_map.view(B, 1, -1)      # (B,1,N)
        z = torch.cat([z, S_map_flat.transpose(1,2)], dim=-1)  # (B,N,513)
        dense_feat = self.out_proj(z.transpose(1,2).view(B,513,H,W))   # project back to 512

        end_time=   time.time()
        print(f"GroundingModel forward time: {end_time - start_time:.4f} sec")

        return dense_feat

def build_grounding_model(args) -> GroundingModel:
    model = GroundingModel(
        d_model=32,
        n_heads=4,
        score_dim=32,
        n_colors=3,
        n_left_ord=6,
        n_right_ord=6,
        grid_hw=(8, 20),
    )
    return model