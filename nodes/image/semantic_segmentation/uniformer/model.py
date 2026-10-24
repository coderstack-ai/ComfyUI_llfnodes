"""Load CondadosAI UniFormer-S ADE20K UPerNet weights and run semantic segmentation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .backbone import UniFormerBackbone
from .uper_head import FCNAuxHead, UPerHead

REPO_ID = "CondadosAI/uniformer_s_ade20k_seg"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class UniFormerSeg(nn.Module):
    def __init__(self, num_classes: int = 150):
        super().__init__()
        self.backbone = UniFormerBackbone()
        self.decode_head = UPerHead(num_classes=num_classes)
        self.auxiliary_head = FCNAuxHead(num_classes=num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.decode_head(feats)
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)


def _load_state_dict(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=True)


def load_uniformer_ade20k(local_dir: str | Path, device: torch.device) -> UniFormerSeg:
    local_dir = Path(local_dir)
    cfg_path = local_dir / "config.json"
    num_classes = 150
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        num_classes = int(cfg.get("num_classes", 150))

    weight_path = local_dir / "model.safetensors"
    if not weight_path.exists():
        cands = list(local_dir.glob("*.safetensors")) + list(local_dir.glob("*.bin")) + list(local_dir.glob("*.pth"))
        if not cands:
            raise FileNotFoundError(f"UniFormer 权重不存在: {local_dir}")
        weight_path = cands[0]

    model = UniFormerSeg(num_classes=num_classes)
    state = _load_state_dict(weight_path)
    # CondadosAI keys are already backbone.* / decode_head.* / auxiliary_head.*
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Ignore BN num_batches_tracked mismatches if any; require no unexpected backbone holes
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing:
        raise RuntimeError(f"UniFormer 权重缺少关键键（示例）: {missing[:8]}")
    model.to(device)
    model.eval()
    return model


def _resize_short_side(pil: Image.Image, short_side: int = 512, long_side_cap: int = 2048) -> Image.Image:
    w, h = pil.size
    scale = short_side / float(min(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    if max(nh, nw) > long_side_cap:
        scale = long_side_cap / float(max(nh, nw))
        nh, nw = int(round(nh * scale)), int(round(nw * scale))
    # keep multiples of 32 for UniFormer strides
    nh = max(32, (nh + 31) // 32 * 32)
    nw = max(32, (nw + 31) // 32 * 32)
    return pil.resize((nw, nh), Image.BILINEAR)


def preprocess(pil: Image.Image, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    orig = (pil.height, pil.width)
    img = _resize_short_side(pil.convert("RGB"))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t, orig


@torch.inference_mode()
def predict_label_map(model: UniFormerSeg, pil: Image.Image, device: torch.device) -> np.ndarray:
    x, (oh, ow) = preprocess(pil, device)
    logits = model(x)
    label = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)
    if label.shape != (oh, ow):
        label_t = torch.from_numpy(label.astype(np.float32))[None, None]
        label_t = F.interpolate(label_t, size=(oh, ow), mode="nearest")
        label = label_t.squeeze().numpy().astype(np.int64)
    return label
