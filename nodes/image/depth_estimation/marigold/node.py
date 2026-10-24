import numpy as np
import torch
from comfy_api.latest import io

import comfy.model_management as model_management

from ....model_store import hf_local
from ..common import (
    CATEGORY,
    batch_depth_to_image,
    cache_get,
    cache_set,
    get_torch_device,
    tensor_to_pils,
)

MODELS = {
    # v1.1 already supports 1–4 steps with DDIM; recommended checkpoint.
    "v1.1": "prs-eth/marigold-depth-v1-1",
    # Hub only has LCM v1-0 (no marigold-depth-lcm-v1-1). Keep both labels for old workflows.
    "lcm-v1.0": "prs-eth/marigold-depth-lcm-v1-0",
    "lcm-v1.1": "prs-eth/marigold-depth-lcm-v1-0",
}


def _load_marigold(model_id: str):
    try:
        from diffusers import MarigoldDepthPipeline
    except Exception as e:
        raise RuntimeError(
            "无法导入 MarigoldDepthPipeline。当前环境需兼容组合："
            "huggingface_hub>=0.34,<1.0 且 diffusers>=0.32,<0.40 "
            "（diffusers 0.40+ 要求 hub>=1.23，会与 transformers 4.x 冲突）。\n"
            f"原始错误: {e}"
        ) from e

    device = get_torch_device()
    key = f"marigold:{model_id}@{device}"
    pipe = cache_get(key)
    if pipe is not None:
        return pipe

    local = hf_local(model_id, folder_name="diffusers")
    model_management.unload_all_models()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = MarigoldDepthPipeline.from_pretrained(
        local, torch_dtype=dtype, local_files_only=True
    )
    pipe = pipe.to(device)
    return cache_set(key, pipe)


class MarigoldNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.Marigold",
            display_name="Marigold",
            category=CATEGORY,
            description="Marigold 扩散深度估计（diffusers）。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="v1.1"),
                io.Int.Input("steps", default=4, min=1, max=50, step=1),
                io.Int.Input("ensemble_size", default=1, min=1, max=10, step=1),
                io.Boolean.Input("invert", default=False, label_on="near=black", label_off="near=white"),
            ],
            outputs=[io.Image.Output("DEPTH")],
        )

    @classmethod
    def execute(cls, image, model, steps, ensemble_size, invert):
        pipe = _load_marigold(MODELS[model])
        depths = []
        for pil in tensor_to_pils(image):
            out = pipe(
                pil,
                num_inference_steps=steps,
                ensemble_size=ensemble_size,
                match_input_resolution=True,
                output_type="np",
            )
            depth = out.prediction
            depth = np.squeeze(np.asarray(depth, dtype=np.float32))
            depths.append(depth)
        return io.NodeOutput(batch_depth_to_image(depths, invert=invert))
