import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import comfy.model_management as model_management

CATEGORY = "llfnodes/图像处理/表面法线估计"

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


def normal_array_to_image(normal: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Accept normals in [-1,1] CHW/HWC or [0,1] visualization HWC.
    Return IMAGE [1,H,W,3] in 0-1 (mapped from [-1,1] if needed).
    """
    if isinstance(normal, torch.Tensor):
        normal = normal.detach().float().cpu().numpy()
    normal = np.asarray(normal, dtype=np.float32)
    normal = np.squeeze(normal)

    if normal.ndim == 3 and normal.shape[0] == 3:
        normal = np.transpose(normal, (1, 2, 0))

    if normal.min() < -1e-3:
        normal = (normal + 1.0) * 0.5
    normal = np.clip(normal, 0.0, 1.0)
    if normal.shape[-1] == 1:
        normal = np.repeat(normal, 3, axis=-1)
    return torch.from_numpy(normal)[None, ...]


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
            "需要安装 comfyui_controlnet_aux（提供 DSINE / NormalBae）。\n"
            "请用 ComfyUI Manager 安装 ControlNet Auxiliary Preprocessors。"
        ) from e


def run_controlnet_normal_detector(detector, image: torch.Tensor, **kwargs) -> torch.Tensor:
    outs = []
    for pil in tensor_to_pils(image):
        arr = np.asarray(pil)
        result = detector(arr, output_type="np", **kwargs)
        if isinstance(result, Image.Image):
            result = np.asarray(result)
        result = np.asarray(result, dtype=np.float32)
        if result.max() > 1.5:
            result = result / 255.0
        outs.append(normal_array_to_image(result))
    return torch.cat(outs, dim=0)


def run_marigold_normals(
    model_id: str,
    image: torch.Tensor,
    steps: int = 4,
    ensemble_size: int = 1,
) -> torch.Tensor:
    try:
        from diffusers import MarigoldNormalsPipeline
    except Exception as e:
        raise RuntimeError(
            "无法导入 MarigoldNormalsPipeline。请升级依赖：\n"
            "pip install -U 'diffusers>=0.40' 'huggingface_hub>=0.34'\n"
            f"原始错误：{e}"
        ) from e

    def loader(device):
        from ...model_store import hf_local

        local = hf_local(model_id, folder_name="diffusers")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        pipe = MarigoldNormalsPipeline.from_pretrained(
            local, torch_dtype=dtype, local_files_only=True
        )
        return pipe.to(device)

    pipe = get_cached_model(f"marigold-normals:{model_id}", loader)

    outs = []
    for pil in tensor_to_pils(image):
        result = pipe(
            pil,
            num_inference_steps=steps,
            ensemble_size=ensemble_size,
            match_input_resolution=True,
            output_type="np",
        )
        normal = np.squeeze(np.asarray(result.prediction, dtype=np.float32))
        outs.append(normal_array_to_image(normal))
    return torch.cat(outs, dim=0)
