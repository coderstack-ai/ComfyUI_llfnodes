import numpy as np
import torch
from PIL import Image

import comfy.model_management as model_management

CATEGORY = "llfnodes/图像处理/深度估计"

_hf_pipe_cache: dict[str, object] = {}
_generic_cache: dict[str, object] = {}


def get_torch_device():
    return model_management.get_torch_device()


def tensor_to_pils(image: torch.Tensor) -> list[Image.Image]:
    """Comfy IMAGE [B,H,W,C] float0-1 -> list of RGB PIL."""
    imgs = []
    for i in range(image.shape[0]):
        arr = (image[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        imgs.append(Image.fromarray(arr, mode="RGB"))
    return imgs


def depth_map_to_image(depth: np.ndarray | torch.Tensor, invert: bool = False) -> torch.Tensor:
    """Single depth map -> IMAGE [1,H,W,3] normalized to 0-1."""
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().float().cpu().numpy()
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.squeeze(depth)
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax > dmin:
        depth = (depth - dmin) / (dmax - dmin)
    else:
        depth = np.zeros_like(depth)
    if invert:
        depth = 1.0 - depth
    t = torch.from_numpy(depth)[None, ..., None].expand(-1, -1, -1, 3).contiguous()
    return t


def batch_depth_to_image(depths: list[np.ndarray | torch.Tensor], invert: bool = False) -> torch.Tensor:
    return torch.cat([depth_map_to_image(d, invert=invert) for d in depths], dim=0)


def get_hf_depth_pipeline(model_id: str):
    from ...model_store import hf_local

    device = get_torch_device()
    local = hf_local(model_id)
    cache_key = f"{local}@{device}"
    pipe = _hf_pipe_cache.get(cache_key)
    if pipe is not None:
        return pipe

    from transformers import pipeline

    model_management.unload_all_models()
    pipe = pipeline(
        task="depth-estimation",
        model=local,
        device=device,
    )
    _hf_pipe_cache.clear()
    _hf_pipe_cache[cache_key] = pipe
    return pipe


def run_hf_depth(model_id: str, image: torch.Tensor, invert: bool = False) -> torch.Tensor:
    pipe = get_hf_depth_pipeline(model_id)
    depths = []
    for pil in tensor_to_pils(image):
        out = pipe(pil)
        if "predicted_depth" in out and out["predicted_depth"] is not None:
            depth = out["predicted_depth"]
            if isinstance(depth, torch.Tensor):
                depth = depth.float().cpu().numpy()
            else:
                depth = np.asarray(depth, dtype=np.float32)
            # resize predicted depth to original image size if needed
            if depth.shape[-2:] != (pil.height, pil.width) and depth.ndim >= 2:
                depth_t = torch.from_numpy(np.squeeze(depth).astype(np.float32))[None, None]
                depth_t = torch.nn.functional.interpolate(
                    depth_t, size=(pil.height, pil.width), mode="bicubic", align_corners=False
                )
                depth = depth_t.squeeze().numpy()
        else:
            depth_img = out["depth"]
            if isinstance(depth_img, Image.Image):
                depth = np.array(depth_img, dtype=np.float32)
            else:
                depth = np.asarray(depth_img, dtype=np.float32)
        depths.append(depth)
    return batch_depth_to_image(depths, invert=invert)


def cache_get(key: str):
    return _generic_cache.get(key)


def cache_set(key: str, value):
    _generic_cache.clear()
    _generic_cache[key] = value
    return value
