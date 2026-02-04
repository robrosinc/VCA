import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TinyBackbone(nn.Module):
    """
    Simple conv backbone that maps (B,3,240,640) -> (B,5,8,20)
    Uses strided convs to downsample (240,640) -> (8,20) i.e., /30 and /32-ish.
    This is intentionally light; adjust if your input sizes differ.
    """
    def __init__(self, out_ch: int = 5):
        super().__init__()
        # Downsampling pipeline:
        # 240x640 -> 120x320 -> 60x160 -> 30x80 -> 15x40 -> 8x20
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),  # 120x320
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 60x160
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 30x80
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 15x40
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 8x20 (since 15->8, 40->20)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, kernel_size=1, stride=1, padding=0),  # (B,out_ch,8,20)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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

        # 1) backbone: (B,3,240,640) -> (B,5,8,20)
        self.backbone = TinyBackbone(out_ch=5)

        # 2) objectness head: (B,5,8,20) -> (B,1,8,20)
        self.obj_head = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # 3) concat pos => (B,7,8,20), then project to (B,d_model,8,20)
        self.proj = nn.Conv2d(5 + 2, d_model, kernel_size=1)

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

    def _get_pos_grid(self, device, dtype) -> torch.Tensor:
        if self._pos_grid.numel() == 0 or self._pos_grid.device != device or self._pos_grid.dtype != dtype:
            self._pos_grid = make_xy_pos_grid(self.H, self.W, device=device, dtype=dtype)  # (1,2,H,W)
        return self._pos_grid

    def forward(
        self,
        img: torch.Tensor,                  # (B,3,240,640)
        color_id: torch.Tensor,             # (B,) int64
        left_ord_id: torch.Tensor,          # (B,) int64
        right_ord_id: torch.Tensor,         # (B,) int64
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B = img.shape[0]
        device = img.device
        dtype = img.dtype

        # 1) backbone feature
        f = self.backbone(img)  # (B,5,H,W) expected (8,20)
        if f.shape[-2:] != (self.H, self.W):
            raise ValueError(f"Backbone output grid {f.shape[-2:]} != expected {(self.H, self.W)}")

        # 2) objectness gating
        m_logits = self.obj_head(f)          # (B,1,H,W)
        m_logits = m_logits.clamp(-10, 10)
        m = torch.sigmoid(m_logits)          # (B,1,H,W)
        f_g = f * m                          # (B,5,H,W)

        # 3) concat position
        pos = self._get_pos_grid(device=device, dtype=dtype).expand(B, -1, -1, -1)  # (B,2,H,W)
        h = torch.cat([f_g, pos], dim=1)     # (B,7,H,W)

        # 4) project to d_model and tokenize
        u = self.proj(h)                     # (B,d_model,H,W)
        z0 = u.flatten(2).transpose(1, 2)    # (B, N, d_model) where N=H*W

        # 5) self-attention block (pre-norm style)
        # Self-attn
        z1 = self.self_attn_ln(z0)
        attn_out, attn_w = self.self_attn(z1, z1, z1, need_weights=False)  # (B,N,d_model)
        z = z0 + attn_out

        # FFN
        z2 = self.self_attn_ff_ln(z)
        z = z + self.self_attn_ff(z2)        # (B,N,d_model)

        # 6) text tokens (no mixing)
        t_color = self.emb_color(color_id)       # (B,d_model)
        t_left = self.emb_left(left_ord_id)      # (B,d_model)
        t_right = self.emb_right(right_ord_id)   # (B,d_model)
        t = torch.stack([t_color, t_left, t_right], dim=1)  # (B,3,d_model)

        # 7) per-token scoring maps
        k = self.k_proj(z)  # (B,N,score_dim)

        maps = []
        for i in range(3):
            q = self.q_proj[i](t[:, i, :])       # (B,score_dim)
            # dot-product score: (B,N)
            s = torch.einsum("bd,bnd->bn", q, k) / math.sqrt(self.score_dim)
            S = s.view(B, 1, self.H, self.W)     # (B,1,H,W)
            maps.append(S)

        S_color, S_left, S_right = maps

        # 8) weighted sum combine (stable)
        w = torch.softmax(self.map_weights, dim=0)  # (3,)
        S_logits = w[0] * S_color + w[1] * S_left + w[2] * S_right  # (B,1,H,W)
        S_logits = S_logits.clamp(-50, 50)

        out = {
            "heatmap_logits": S_logits,  # (B,1,8,20)
        }

        if return_aux:
            out.update({
                "mask_logits": m_logits,  # (B,1,8,20)
                "mask_prob": m,           # (B,1,8,20)
                "map_color": S_color,
                "map_left": S_left,
                "map_right": S_right,
                "combine_weights": w.detach().clone(),
            })

        return out

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