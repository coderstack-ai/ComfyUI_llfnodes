import sys
from types import ModuleType

import numpy as np
import torch
from comfy_api.latest import io

import comfy.model_management as model_management

from ..common import CATEGORY, batch_depth_to_image, cache_get, cache_set, get_torch_device, tensor_to_pils
from ....model_store import hf_local

MODELS = {
    "Small": "depth-anything/DA3-SMALL",
    "Base": "depth-anything/DA3-BASE",
    "Large": "depth-anything/DA3-LARGE",
}

# Optional DA3 deps pulled in by export helpers; not required for monocular depth.
_OPTIONAL_STUBS = ("pycolmap", "open3d", "gsplat", "xformers")


def _ensure_optional_stubs():
    for name in _OPTIONAL_STUBS:
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                sys.modules[name] = ModuleType(name)


def _import_da3():
    _ensure_optional_stubs()
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as e:
        raise RuntimeError(
            "Depth Anything V3 需要安装官方包：\n"
            "pip install depth-anything-3 moviepy==1.0.3 e3nn einops omegaconf\n"
            f"原始错误: {e}"
        ) from e
    return DepthAnything3


def _load_da3(model_id: str):
    DepthAnything3 = _import_da3()

    device = get_torch_device()
    key = f"da3:{model_id}@{device}"
    model = cache_get(key)
    if model is not None:
        return model

    model_management.unload_all_models()
    local = hf_local(model_id)
    model = DepthAnything3.from_pretrained(local)
    model = model.to(device=device)
    model.eval()
    return cache_set(key, model)


class DepthAnythingV3Node(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DepthAnythingV3",
            display_name="Depth Anything V3",
            category=CATEGORY,
            description="Depth Anything V3 深度估计（depth-anything-3）。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="Base"),
                io.Boolean.Input("invert", default=False, label_on="near=black", label_off="near=white"),
            ],
            outputs=[io.Image.Output("DEPTH")],
        )

    @classmethod
    def execute(cls, image, model, invert):
        da3 = _load_da3(MODELS[model])
        pils = tensor_to_pils(image)
        prediction = da3.inference(pils)
        depth = prediction.depth
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().float().cpu().numpy()
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 2:
            depths = [depth]
        else:
            depths = [depth[i] for i in range(depth.shape[0])]
        return io.NodeOutput(batch_depth_to_image(depths, invert=invert))
