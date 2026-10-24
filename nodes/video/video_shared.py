"""Shared helpers for video CV nodes."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import torch
from comfy_api.latest import InputImpl
from comfy_api.latest._util import VideoComponents

import comfy.model_management as model_management

_model_cache: dict[str, object] = {}


def get_torch_device():
    return model_management.get_torch_device()


def get_cached_model(cache_key: str, loader):
    device = get_torch_device()
    key = f"{cache_key}@{device}"
    cached = _model_cache.get(key)
    if cached is not None:
        return cached
    model_management.unload_all_models()
    value = loader(device)
    _model_cache.clear()
    _model_cache[key] = value
    return value


def require_pkg(pkg: str, install_hint: str):
    try:
        return __import__(pkg)
    except ImportError as e:
        raise RuntimeError(f"缺少依赖 {pkg}。\n安装：{install_hint}") from e


def video_to_frames(video) -> torch.Tensor:
    components = video.get_components()
    frames = components.images
    if frames.dtype != torch.float32:
        frames = frames.float()
        if frames.max() > 1.5:
            frames = frames / 255.0
    return frames.clamp(0, 1)


def video_meta(video):
    components = video.get_components()
    return components.frame_rate, components.audio, components.metadata


def frames_to_video(frames: torch.Tensor, video=None, fps: float | None = None):
    """frames [T,H,W,C] float -> VideoInput."""
    if frames.dtype != torch.float32:
        frames = frames.float()
        if frames.max() > 1.5:
            frames = frames / 255.0
    audio = None
    metadata = None
    frame_rate = Fraction(int(round((fps or 24) * 1000)), 1000)
    if video is not None:
        c = video.get_components()
        audio = c.audio
        metadata = c.metadata
        if fps is None:
            frame_rate = c.frame_rate
    return InputImpl.VideoFromComponents(
        VideoComponents(images=frames.contiguous(), frame_rate=frame_rate, audio=audio, metadata=metadata)
    )


def array_batch_to_image(arrs) -> torch.Tensor:
    tensors = []
    for arr in arrs:
        a = np.asarray(arr, dtype=np.float32)
        if a.max() > 1.5:
            a = a / 255.0
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=-1)
        tensors.append(torch.from_numpy(np.clip(a, 0, 1))[None, ...])
    return torch.cat(tensors, dim=0)


def draw_boxes_on_frames(frames: torch.Tensor, tracks_per_frame: list[list[tuple]]) -> torch.Tensor:
    """tracks_per_frame[i] = list of (x1,y1,x2,y2,label,color)."""
    import cv2

    outs = []
    for i in range(frames.shape[0]):
        img = (frames[i].detach().cpu().numpy() * 255).astype(np.uint8).copy()
        for item in tracks_per_frame[i] if i < len(tracks_per_frame) else []:
            x1, y1, x2, y2, label, color = item
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            if label:
                cv2.putText(img, str(label), (int(x1), max(0, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        outs.append(img)
    return array_batch_to_image(outs)


def missing_backend(name: str, hint: str):
    raise RuntimeError(f"{name} 需要额外依赖/权重。\n{hint}")
