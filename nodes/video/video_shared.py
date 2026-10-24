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


def _info(name: str, detail: str) -> str:
    return f"{name}: 官方后端未安装，已用本地回退 — {detail}"


def fallback_passthrough(video, name: str):
    frames = video_to_frames(video)
    return frames_to_video(frames, video), _info(name, f"原样返回（T={frames.shape[0]}）")


def fallback_interp_linear(video, name: str):
    frames = video_to_frames(video)
    if frames.shape[0] < 2:
        return fallback_passthrough(video, name)
    outs = []
    for i in range(frames.shape[0] - 1):
        outs.append(frames[i])
        outs.append((frames[i] + frames[i + 1]) * 0.5)
    outs.append(frames[-1])
    return frames_to_video(torch.stack(outs, dim=0), video), _info(name, "线性插帧 2x")


def fallback_denoise_temporal(video, name: str):
    frames = video_to_frames(video)
    t = frames.shape[0]
    outs = []
    for i in range(t):
        ids = [j for j in (i - 1, i, i + 1) if 0 <= j < t]
        outs.append(frames[ids].mean(dim=0))
    return frames_to_video(torch.stack(outs, dim=0), video), _info(name, "±1 帧时间均值")


def fallback_deblur(video, name: str):
    import cv2

    frames = video_to_frames(video)
    outs = []
    for i in range(frames.shape[0]):
        img = (frames[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        blur = cv2.GaussianBlur(img, (0, 0), 1.2)
        sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
        outs.append(sharp)
    return frames_to_video(array_batch_to_image(outs), video), _info(name, "Unsharp 回退")


def fallback_sr(video, name: str, scale: int = 2):
    frames = video_to_frames(video)
    up = torch.nn.functional.interpolate(
        frames.permute(0, 3, 1, 2),
        scale_factor=float(scale),
        mode="bicubic",
        align_corners=False,
    )
    up = up.permute(0, 2, 3, 1).clamp(0, 1)
    return frames_to_video(up, video), _info(name, f"bicubic {scale}x")


def fallback_motion_overlay(video, name: str):
    frames = video_to_frames(video)
    outs = [frames[0]]
    for i in range(1, frames.shape[0]):
        diff = (frames[i] - frames[i - 1]).abs().mean(dim=-1, keepdim=True)
        mask = (diff > 0.04).float()
        color = torch.tensor([1.0, 0.25, 0.25], dtype=frames.dtype, device=frames.device).view(1, 1, 3)
        outs.append(frames[i] * (1.0 - 0.5 * mask) + color * (0.5 * mask))
    return frames_to_video(torch.stack(outs, dim=0), video), _info(name, "帧差运动高亮")


def fallback_depth_video(video, name: str):
    from ..image.depth_estimation.common import run_hf_depth

    frames = video_to_frames(video)
    depth = run_hf_depth("depth-anything/Depth-Anything-V2-Small-hf", frames, invert=False)
    return frames_to_video(depth, video), _info(name, "逐帧 Depth Anything V2")


def fallback_keyframes(video, name: str):
    frames = video_to_frames(video)
    kept = [0]
    prev = frames[0].mean(dim=-1)
    for i in range(1, frames.shape[0]):
        cur = frames[i].mean(dim=-1)
        if float((cur - prev).abs().mean()) > 0.08:
            kept.append(i)
            prev = cur
    if len(kept) == 1 and frames.shape[0] > 1:
        kept = torch.linspace(0, frames.shape[0] - 1, steps=min(8, frames.shape[0])).long().tolist()
    out = frames[kept]
    return frames_to_video(out, video), _info(name, f"直方图切镜，保留 {out.shape[0]} 帧")


def fallback_understanding(video, name: str):
    frames = video_to_frames(video)
    return (
        frames_to_video(frames, video),
        _info(name, f"请改用图像 Qwen-VL/Florence2；已原样返回视频 T={frames.shape[0]}"),
    )


def fallback_track_yolo(video, name: str):
    try:
        from ultralytics import YOLO

        from ..model_store import ensure_yolo_weight
    except Exception:
        return fallback_passthrough(video, name)

    frames = video_to_frames(video)
    weight = ensure_yolo_weight("yolov8n.pt")
    model = YOLO(str(weight))
    tracks = []
    outs = []
    import cv2

    for i in range(frames.shape[0]):
        img = (frames[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        res = model.track(bgr, persist=True, verbose=False)[0]
        canvas = img.copy()
        line = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy() if res.boxes.id is not None else [None] * len(xyxy)
            for box, tid in zip(xyxy, ids):
                x1, y1, x2, y2 = map(int, box)
                label = f"id{int(tid)}" if tid is not None else "obj"
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(canvas, label, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                line.append(f"{label}:{x1},{y1},{x2},{y2}")
        outs.append(canvas)
        tracks.append("; ".join(line) if line else "(none)")
    return (
        frames_to_video(array_batch_to_image(outs), video),
        _info(name, "YOLO track 回退") + "\n" + "\n".join(tracks[:50]),
    )
