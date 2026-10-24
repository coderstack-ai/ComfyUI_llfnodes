import cv2
import numpy as np
import torch
from comfy_api.latest import io

from ...cv_shared import array_to_image, tensor_to_pils

CATEGORY = "llfnodes/图像处理/光照估计"


def _retinex_albedo_shading(rgb: np.ndarray):
    h, w = rgb.shape[:2]
    lum = rgb.mean(axis=-1).astype(np.float32)
    sigma = max(4.0, min(h, w) / 24.0)
    shading = cv2.GaussianBlur(lum, (0, 0), sigmaX=sigma)
    shading = np.clip(shading, 1e-3, 1.0)
    albedo = np.clip(rgb / shading[..., None], 0.0, 1.0)
    # Keep albedo in a displayable range (chromaticity can exceed 1 on dark pixels)
    a_max = float(np.percentile(albedo, 99))
    if a_max > 1e-6:
        albedo = np.clip(albedo / max(a_max, 1.0), 0.0, 1.0)
    return albedo.astype(np.float32), shading.astype(np.float32)


class InverseRenderingNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.InverseRendering",
            display_name="逆渲染模型",
            category=CATEGORY,
            description="Retinex 逆渲染近似：分解 albedo/shading，并用翻转光照做重打光预览。",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("RELIGHT_PREVIEW"), io.String.Output("PARAMS")],
        )

    @classmethod
    def execute(cls, image):
        outs, params = [], []
        for pil in tensor_to_pils(image):
            rgb = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
            albedo, shading = _retinex_albedo_shading(rgb)
            amb = float(np.percentile(shading, 10))
            dif = float(np.clip(np.percentile(shading, 90) - amb, 1e-3, 1.0))
            # Flip directional shading so relight is clearly different from the input.
            t = np.clip((shading - amb) / dif, 0.0, 1.0)
            new_shading = amb + dif * (1.0 - t)
            preview = np.clip(albedo * new_shading[..., None], 0.0, 1.0)
            outs.append(array_to_image(preview))
            params.append(f"ambient={amb:.4f}, directional={dif:.4f}")
        return io.NodeOutput(torch.cat(outs, dim=0), "\n".join(params))
