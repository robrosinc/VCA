#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import zlib
import io
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import h5py
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from detr.util.misc import NestedTensor
from tqdm import tqdm
import wandb

from detr.models.backbone  import Resnet17, build_backbone
from detr.models.grounding_model import GroundingModel

# ----------------------------
# Utilities
# ----------------------------
def decode_image_to_chw_uint8(img_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(img_bytes)) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)  # HWC
    return np.transpose(arr, (2, 0, 1))  # CHW

def to_nested(img):
    mask = torch.zeros(
        (img.shape[0], img.shape[2], img.shape[3]),
        dtype=torch.bool,
        device=img.device,
    )
    return NestedTensor(img, mask)

def crop_and_resize_chw(
    chw: np.ndarray,
    crop: Tuple[int, int, int, int],  # (y0, y1, x0, x1)
    out_hw: Tuple[int, int],          # (H, W)
) -> torch.Tensor:
    c, h, w = chw.shape
    y0, y1, x0, x1 = crop
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    cropped = chw[:, y0:y1, x0:x1].copy()

    t = torch.from_numpy(cropped).float() / 255.0  # (3,Hc,Wc)
    t = t.unsqueeze(0)
    t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return t.squeeze(0)

from typing import Tuple
import numpy as np

def decode_numeric2(vec) -> Tuple[int, int, int]:
    """
    vec: array-like of shape (14,)
    returns: (d1, d2, d3)
    """
    vec = np.asarray(vec)

    if vec.shape != (14,):
        raise ValueError(f"numeric2 must have shape (14,), got {vec.shape}")

    d1 = d2 = d3 = 0

    # first digit
    if vec[0] == 1:
        d1 = 1
    elif vec[1] == 1:
        d1 = 2

    # second digit
    for i in range(2, 8):
        if vec[i] == 1:
            d2 = i - 2
            break

    # third digit
    for i in range(8, 14):
        if vec[i] == 1:
            d3 = i - 8
            break

    return d1, d2, d3


def decompress_mask_zlib(mask_bytes: bytes, h: int = 480, w: int = 640) -> np.ndarray:
    raw = zlib.decompress(mask_bytes)
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size != h * w:
        raise ValueError(f"Mask size mismatch: got {arr.size}, expected {h*w}")
    mask = arr.reshape(h, w)
    # enforce binary 0/1 (in case it comes as 0/255)
    mask = (mask > 0).astype(np.uint8)
    return mask


# ----------------------------
# Dataset
# ----------------------------
@dataclass
class SampleIndex:
    path: str
    t: int


class H5PromptDataset(Dataset):
    """
    mode:
      - "transition": prev == 000 and cur != 000
      - "nonzero":    cur != 000
    """
    def __init__(
        self,
        h5_paths: List[str],
        crop: Tuple[int, int, int, int] = (0, 480, 0, 640),
        out_hw: Tuple[int, int] = (240, 640),
        mode: str = "transition",
    ):
        super().__init__()
        assert mode in ("transition", "nonzero")
        self.h5_paths = h5_paths
        self.crop = crop
        self.out_hw = out_hw
        self.mode = mode
        self.indices: List[SampleIndex] = []
        self._build_index()

    def _build_index(self):
        threshold = 50  # e.g. 50
        self.indices = []

        for p in self.h5_paths:
            with h5py.File(p, "r") as f:
                masks = f["prompts/masks/head_camera"]
                T = masks.shape[0]

                prev_count = 0

                for t in range(T):
                    mask_bytes = masks[t]

                    if isinstance(mask_bytes, np.ndarray):
                        mask_bytes = mask_bytes.tobytes()

                    mask = decompress_mask_zlib(mask_bytes, 480, 640)  # (H,W), 0/1
                    curr_count = int(mask.sum())

                    # transition: empty -> active
                    if prev_count <= threshold and curr_count > threshold:
                        self.indices.append(SampleIndex(path=p, t=int(t)))

                    prev_count = curr_count

        if len(self.indices) == 0:
            raise RuntimeError(
                "No training frames found (check mask threshold or mask decoding)."
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.indices[idx]

        with h5py.File(item.path, "r") as f:
            # ---- image ----
            img_bytes = f["observations/images/head_camera"][item.t]
            if isinstance(img_bytes, np.ndarray):
                img_bytes = img_bytes.tobytes()
            chw = decode_image_to_chw_uint8(img_bytes)  # (3,480,1280)

            # ---- prompt ----
            numeric2_vec = f["prompts/numeric2"][item.t]   # (14,)
            n1, n2, n3 = decode_numeric2(numeric2_vec)

            # ---- mask ----
            mask_bytes = f["prompts/masks/head_camera"][item.t]
            if isinstance(mask_bytes, np.ndarray):
                mask_bytes = mask_bytes.tobytes()

            mask = decompress_mask_zlib(mask_bytes, 480, 640)  # (480,640), {0,1}

        # ---- resize image ----
        img_t = crop_and_resize_chw(
            chw,
            crop=self.crop,
            out_hw=self.out_hw,   # (240,640)
        )  # (3,240,640)

        # ---- resize mask for teacher ----
        mask_t = torch.from_numpy(mask).float()            # (480,640)
        mask_t = mask_t.unsqueeze(0).unsqueeze(0)          # (1,1,480,640)
        mask_t = F.interpolate(
            mask_t,
            size=(240, 640),
            mode="nearest",      # KEEP BINARY
        ).squeeze(0)             # (1,240,640)

        prompt_t = torch.tensor([n1, n2, n3], dtype=torch.long)

        return {
            "image": img_t,       # (3,240,640)
            "prompt": prompt_t,   # (3,)
            "mask": mask_t,       # (1,240,640)
        }


# ----------------------------
# Partial checkpoint loading
# ----------------------------
def extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        return ckpt.get("model_state", ckpt)
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise KeyError("Could not find state_dict in checkpoint.")


def load_partial_state_dict(module: nn.Module, ckpt_state: Dict[str, torch.Tensor], prefixes):
    own = module.state_dict()
    filtered = {}
    for k, v in ckpt_state.items():
        for p in prefixes:
            if k.startswith(p):
                new_k = k[len(p):]
                if new_k in own and own[new_k].shape == v.shape:
                    filtered[new_k] = v
                break
    missing, unexpected = module.load_state_dict(filtered, strict=False)

    print(f"[teacher load] loaded: {len(filtered)} tensors")
    print(f"[teacher load] missing: {len(missing)}")
    print(f"[teacher load] unexpected: {len(unexpected)}")
    num_teacher_params = sum(1 for _ in module.state_dict())
    print(f"[teacher] total params in model: {num_teacher_params}")
    print(f"[partial_load] loaded {len(filtered)} tensors into teacher")


def load_teacher_from_ckpt(teacher: nn.Module, ckpt_state: Dict[str, torch.Tensor]):
    own = teacher.state_dict()

    loaded = {}
    skipped_shape = []
    skipped_name = []

    for k, v in ckpt_state.items():
        new_k = None

        # Case 1: already matches Resnet17
        if k in own:
            new_k = k

        # Case 2: DETR mask backbone
        elif k.startswith("model.mask_backbones.0.0."):
            new_k = k.replace("model.mask_backbones.0.0.", "", 1)

        if new_k is not None:
            if new_k in own and own[new_k].shape == v.shape:
                loaded[new_k] = v
            else:
                skipped_shape.append((k, v.shape, own.get(new_k, None)))

    missing = set(own.keys()) - set(loaded.keys())

    print(f"[teacher load] loaded: {len(loaded)}")
    print(f"[teacher load] missing: {len(missing)}")
    print(f"[teacher load] skipped (shape mismatch): {len(skipped_shape)}")

    teacher.load_state_dict(loaded, strict=False)

def load_backbone(backbone: nn.Module, ckpt_state: dict):
    own = backbone.state_dict()
    loaded = {}

    prefix = "model.backbones.0."

    for k, v in ckpt_state.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
            if new_k in own and own[new_k].shape == v.shape:
                loaded[new_k] = v

    missing, unexpected = backbone.load_state_dict(loaded, strict=False)

    print(f"[backbone] loaded {len(loaded)} tensors")
    print(f"[backbone] missing {len(missing)} unexpected {len(unexpected)}")

# ----------------------------
# Losses
# ----------------------------
def distill_kl_softmax(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    # (B,512,1,8,20) -> (N,160)
    s = student_logits.flatten(start_dim=3).reshape(-1, 160)
    t = teacher_logits.flatten(start_dim=3).reshape(-1, 160)
    log_p_s = F.log_softmax(s / T, dim=-1)
    p_t = F.softmax(t / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


def distill_bce_sigmoid(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    target = torch.sigmoid(teacher_logits).detach()
    return F.binary_cross_entropy_with_logits(student_logits, target)

def distill_mse(student_feat: torch.Tensor,
                teacher_feat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_feat, teacher_feat.detach())

# ----------------------------
# Val
# ----------------------------
def validate(student, teacher, val_loader, device):
    student.eval()
    teacher.eval()

    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device, non_blocking=True)
            prompt = batch["prompt"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            t_feat = teacher(mask)["layer4"]                # (B,512,8,20)
            s_feat = student(
                img,
                prompt[:,0],
                prompt[:,1],
                prompt[:,2],
            )["dense_features"]                              # same shape

            loss = F.mse_loss(s_feat, t_feat)
            total_loss += loss.item()
            n += 1

    return total_loss / max(n, 1)

# ----------------------------
# Train
# ----------------------------
def main(args):
    all_h5 = sorted(glob.glob(args.data_glob))
    if not all_h5:
        raise FileNotFoundError(f"No files match: {args.data_glob}")

    split = int(0.9 * len(all_h5))
    train_h5 = all_h5[:split]
    val_h5   = all_h5[split:]

    print(f"[data] train files: {len(train_h5)}, val files: {len(val_h5)}")
    train_ds = H5PromptDataset(train_h5, mode=args.mode)
    val_ds   = H5PromptDataset(val_h5,   mode=args.mode)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,          # shuffle ONLY training
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,         # NEVER shuffle validation
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Dataset size: {len(train_ds)} samples from {len(train_h5)} files.")
    print(f"Validation size: {len(val_ds)} samples from {len(val_h5)} files.")

    device = torch.device(args.device)
    model = GroundingModel(
        d_model=512,
        n_heads=16,
        score_dim=32,
        n_colors=3,
        n_left_ord=6,
        n_right_ord=6,
        grid_hw=(8, 20),
    )
    student = model.to(device)
    teacher = Resnet17().to(device).eval()

    backbone = build_backbone(args).to(device).eval()

    # print("\n[teacher model keys sample]")
    # for i, k in enumerate(backbone.state_dict().keys()):
    #     print(k)
    #     if i > 30:
    #         break

    ckpt = torch.load(args.policy_ckpt, map_location="cpu")
    state = extract_state_dict(ckpt)
    # for k in state.keys():
    #     if k.startswith("model.backbones"):
    #         print(k)
    load_teacher_from_ckpt(teacher, state)
    load_backbone(backbone, state)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    wandb.init(
        project="mask_teacher_distill",
        reinit=True,
        entity="donggunkim-kyung-hee-university",
        name="mask_teacher_distill",
    )

    global_step = 0
    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        student.train()
        train_loss = 0.0
        for batch in tqdm(train_dl, desc=f"[train] epoch {ep}", dynamic_ncols=True):
            img = batch["image"].to(device, non_blocking=True)      # (B,3,240,640)
            prompt = batch["prompt"].to(device, non_blocking=True)  # (B,3)
            mask = batch["mask"].to(device, non_blocking=True)      # (B,1,480,640)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.amp):
                    image_feats, image_pos = backbone(img)

                    feat = image_feats[0]
                    t_logits = teacher(mask)["layer4"].squeeze(2)
            with torch.cuda.amp.autocast(enabled=args.amp):
                s_logits = student(feat, image_pos[0], prompt[:,0], prompt[:,1], prompt[:,2])
                loss = distill_mse(s_logits, t_logits.detach())

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item()
            global_step += 1

        train_loss /= len(train_dl)

        # ===== VALIDATION =====
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"[val] epoch {ep}", dynamic_ncols=True):
                img    = batch["image"].to(device, non_blocking=True)
                mask   = batch["mask"].to(device, non_blocking=True)
                prompt = batch["prompt"].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=args.amp):
                    t_feat = teacher(mask)["layer4"].squeeze(2)
                    img_feats, image_pos = backbone(img)
                    feat = img_feats[0]
                    pos = image_pos[0]
                    s_feat = student(feat, pos, prompt[:,0], prompt[:,1], prompt[:,2])
                    loss = F.mse_loss(s_feat, t_feat)

                val_loss += loss.item()

        val_loss /= len(val_dl)

        wandb.log({
            "train/epoch_loss": train_loss,
            "val/loss": val_loss,
            "train/lr": opt.param_groups[0]["lr"],
        }, step=global_step)

        print(f"[epoch {ep}] train={train_loss:.6f}  val={val_loss:.6f}")

        # ===== checkpoint best =====
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(args.out_dir, exist_ok=True)
            path = os.path.join(args.out_dir, "student_best.pt")
            torch.save({
                "state_dict": student.state_dict(),
                "epoch": ep,
                "val_loss": val_loss,
            }, path)
            print(f"[checkpoint] saved best model → {path}")

    wandb.finish()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", type=str, default="/home/robros/labelmaker_test/results/*.hdf5")
    ap.add_argument("--policy_ckpt", type=str, default="/home/robros/labelmaker_test/policy_step_400000_seed_10.ckpt")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--mode", type=str, default="transition", choices=["transition", "nonzero"])
    ap.add_argument("--out_dir", type=str, default="./outputs")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--masks", action="store_true")
    ap.add_argument("--backbone", type=str, default="resnet34")
    ap.add_argument("--dilation", action="store_true")
    ap.add_argument("--position_embedding", type=str, default="sine")                
    ap.add_argument("--hidden_dim", type=int, default=512)
                                    

    ap.add_argument("--loss_mix_kl", type=float, default=0.7)
    ap.add_argument("--temp", type=float, default=2.0)

    # IMPORTANT: adjust these once you inspect checkpoint keys
    ap.add_argument("--teacher_prefixes", nargs="+",
                    default=["model.mask_backbones.0.0, model.backbones.0"],
                    help="Prefixes in checkpoint for teacher weights")

    args = ap.parse_args()
    main(args)
