import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import comfy.model_management as model_management

CATEGORY = "llfnodes/图像处理/边缘与线稿提取"

_model_cache: dict[str, object] = {}


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


def array_to_image(arr: np.ndarray | torch.Tensor | Image.Image) -> torch.Tensor:
    if isinstance(arr, Image.Image):
        arr = np.asarray(arr)
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().float().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[..., None]
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

    custom_nodes = Path(__file__).resolve().parents[4]
    aux_src = custom_nodes / "comfyui_controlnet_aux" / "src"
    if aux_src.is_dir() and str(aux_src) not in sys.path:
        sys.path.insert(0, str(aux_src))
    try:
        import custom_controlnet_aux  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "需要安装 comfyui_controlnet_aux（提供 HED / PiDiNet / TEED / AnyLine）。\n"
            "请用 ComfyUI Manager 安装 ControlNet Auxiliary Preprocessors。"
        ) from e


def run_controlnet_detector(detector, image: torch.Tensor, **kwargs) -> torch.Tensor:
    outs = []
    for pil in tensor_to_pils(image):
        result = detector(np.asarray(pil), output_type="np", **kwargs)
        outs.append(array_to_image(result))
    return torch.cat(outs, dim=0)
