import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import comfy.model_management as model_management

_model_cache: dict[str, object] = {}

_PALETTE = np.array(
    [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
        [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
        [128, 0, 128], [0, 128, 128], [255, 128, 0], [255, 0, 128], [128, 255, 0],
        [0, 255, 128], [128, 0, 255], [0, 128, 255], [255, 128, 128], [128, 255, 128],
    ],
    dtype=np.uint8,
)


def get_torch_device():
    return model_management.get_torch_device()


def tensor_to_pils(image: torch.Tensor) -> list[Image.Image]:
    imgs = []
    for i in range(image.shape[0]):
        arr = (image[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        imgs.append(Image.fromarray(arr, mode="RGB"))
    return imgs


def pils_to_tensor(pils: list[Image.Image]) -> torch.Tensor:
    tensors = []
    for pil in pils:
        arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr)[None, ...])
    return torch.cat(tensors, dim=0)


def array_to_image(arr) -> torch.Tensor:
    if isinstance(arr, Image.Image):
        arr = np.asarray(arr)
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().float().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return torch.from_numpy(np.clip(arr, 0.0, 1.0))[None, ...]


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


def ensure_controlnet_aux():
    try:
        import custom_controlnet_aux  # noqa: F401
        return
    except ImportError:
        pass
    custom_nodes = Path(__file__).resolve().parents[3]
    aux_src = custom_nodes / "comfyui_controlnet_aux" / "src"
    if aux_src.is_dir() and str(aux_src) not in sys.path:
        sys.path.insert(0, str(aux_src))
    try:
        import custom_controlnet_aux  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "需要安装 comfyui_controlnet_aux。请用 ComfyUI Manager 安装 ControlNet Auxiliary Preprocessors。"
        ) from e


def run_controlnet_detector(detector, image: torch.Tensor, **kwargs) -> torch.Tensor:
    outs = []
    for pil in tensor_to_pils(image):
        result = detector(np.asarray(pil), output_type="np", **kwargs)
        if isinstance(result, tuple):
            result = result[0]
        outs.append(array_to_image(result))
    return torch.cat(outs, dim=0)


def draw_boxes(pil: Image.Image, boxes, labels=None, scores=None) -> Image.Image:
    out = pil.copy()
    draw = ImageDraw.Draw(out)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        color = tuple(int(c) for c in _PALETTE[i % len(_PALETTE)])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = ""
        if labels is not None:
            text = str(labels[i])
        if scores is not None:
            text = f"{text} {float(scores[i]):.2f}".strip()
        if text:
            draw.text((x1 + 2, max(0, y1 - 12)), text, fill=color)
    return out


def overlay_mask(pil: Image.Image, mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay a soft mask or integer label/panoptic map onto ``pil``.

    Soft masks: values in ``[0, 1]``.
    Label / panoptic maps: integer segment ids — never divide by 255.
    """
    base = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    h, w = base.shape[:2]
    mask = np.squeeze(np.asarray(mask))
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

    color = np.zeros((h, w, 3), dtype=np.float32)
    mx = float(np.nanmax(mask)) if mask.size else 0.0
    # Soft probability / binary mask
    if mx <= 1.0 + 1e-3:
        soft = mask.astype(np.float32)
        has = soft > 0.5
        color[has] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        # Panoptic / semantic id map (ids can be 1..N or large segment ids)
        labels = np.rint(mask).astype(np.int64)
        for idx, cls_id in enumerate(np.unique(labels)):
            cid = int(cls_id)
            if cid < 0:
                continue
            color[labels == cid] = _PALETTE[idx % len(_PALETTE)].astype(np.float32) / 255.0
        has = color.sum(axis=-1) > 0

    blended = base.copy()
    blended[has] = base[has] * (1.0 - float(alpha)) + color[has] * float(alpha)
    return Image.fromarray((blended * 255.0).clip(0, 255).astype(np.uint8))


def require_pkg(pkg: str, install_hint: str):
    try:
        return __import__(pkg)
    except ImportError as e:
        raise RuntimeError(f"缺少依赖 {pkg}。\n安装：{install_hint}") from e
