from pathlib import Path

import folder_paths
import numpy as np
import torch
from comfy_api.latest import io
from PIL import Image

from ....model_store import ensure_hf_file
from ..common import (
    CATEGORY,
    batch_depth_to_image,
    cache_get,
    cache_set,
    get_torch_device,
    tensor_to_pils,
)

# Giant2 ONNX export is broken (Add_6 int64/float type mismatch). Do not offer it.
MODELS = {
    "ViT-Small": ("onnx-community/metric3d-vit-small", "onnx/model.onnx"),
    "ViT-Large": ("onnx-community/metric3d-vit-large", "onnx/model.onnx"),
}

# Metric3D canonical inference size; H/W must be divisible by 14*2=28 for ViT.
_CANONICAL = (616, 1064)  # H, W


def _pad_to_multiple(h: int, w: int, multiple: int = 28) -> tuple[int, int]:
    return ((h + multiple - 1) // multiple) * multiple, ((w + multiple - 1) // multiple) * multiple


def _load_session(repo_id: str, filename: str):
    import onnxruntime as ort

    device = get_torch_device()
    key = f"metric3d-onnx:{repo_id}:{filename}@{device}"
    sess = cache_get(key)
    if sess is not None:
        return sess

    local = Path(folder_paths.models_dir) / "llfnodes" / "metric3d" / Path(filename).name
    model_path = str(ensure_hf_file(repo_id, filename, local))
    providers = ["CPUExecutionProvider"]
    if device.type == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        raise RuntimeError(
            f"无法加载 Metric3D ONNX（{repo_id}）。"
            "请改用 ViT-Small / ViT-Large；ViT-Giant2 的社区 ONNX 导出存在类型错误。"
            f"\n原始错误: {e}"
        ) from e
    return cache_set(key, sess)


def _run_metric3d(sess, pil: Image.Image) -> np.ndarray:
    # Export wrapper expects float32 RGB in [0,255] NCHW; it normalizes internally.
    rgb = np.asarray(pil.convert("RGB"), dtype=np.float32)
    oh, ow = rgb.shape[:2]
    th, tw = _CANONICAL
    # Keep aspect, then pad to multiple of 28
    scale = min(th / oh, tw / ow)
    nh, nw = max(28, int(round(oh * scale))), max(28, int(round(ow * scale)))
    nh, nw = _pad_to_multiple(nh, nw, 28)
    resized = np.asarray(pil.convert("RGB").resize((nw, nh), Image.BICUBIC), dtype=np.float32)
    inp = resized.transpose(2, 0, 1)[None, ...]  # 1,3,H,W
    outputs = sess.run(None, {"pixel_values": inp})
    depth = np.squeeze(np.asarray(outputs[0], dtype=np.float32))
    if depth.shape != (oh, ow):
        depth_t = torch.from_numpy(depth)[None, None]
        depth_t = torch.nn.functional.interpolate(
            depth_t, size=(oh, ow), mode="bilinear", align_corners=False
        )
        depth = depth_t.squeeze().numpy()
    return depth


class Metric3DNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.Metric3D",
            display_name="Metric3D",
            category=CATEGORY,
            description="Metric3D 度量深度（ONNX）。ViT-Giant2 社区导出损坏，已移除。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="ViT-Small"),
                io.Boolean.Input("invert", default=False, label_on="near=black", label_off="near=white"),
            ],
            outputs=[io.Image.Output("DEPTH")],
        )

    @classmethod
    def execute(cls, image, model, invert):
        # Old workflows may still pass ViT-Giant2
        if model not in MODELS:
            raise RuntimeError(
                f"不支持的 Metric3D 模型: {model}。"
                "onnx-community/metric3d-vit-giant2 的 ONNX 存在 Add 节点类型错误，无法加载。"
                "请改用 ViT-Small 或 ViT-Large。"
            )
        repo_id, filename = MODELS[model]
        sess = _load_session(repo_id, filename)
        depths = [_run_metric3d(sess, pil) for pil in tensor_to_pils(image)]
        return io.NodeOutput(batch_depth_to_image(depths, invert=invert))
