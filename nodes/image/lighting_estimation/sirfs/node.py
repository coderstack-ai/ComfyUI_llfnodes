import cv2
import numpy as np
import torch
from comfy_api.latest import io

from ...cv_shared import array_to_image, tensor_to_pils

CATEGORY = "llfnodes/图像处理/光照估计"


def _estimate_light_and_shading(rgb: np.ndarray):
    """Classical SfS-lite: light from intensity gradients, shading = N·L with scaled slopes."""
    gray = rgb.mean(axis=-1).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    gy, gx = np.gradient(blur)

    lx = float(-gx.mean())
    ly = float(-gy.mean())
    nxy = float(np.hypot(lx, ly))
    # Prefer a tilted light; pure frontal (0,0,1) yields near-white N·L maps.
    tilt = 0.65
    if nxy < 1e-6:
        lx, ly = tilt * 0.6, -tilt * 0.4
    else:
        lx, ly = lx / nxy * tilt, ly / nxy * tilt
    lz = float(np.sqrt(max(1e-6, 1.0 - lx * lx - ly * ly)))

    # Image gradients are tiny; scale slopes so normals actually tilt.
    p90 = float(np.percentile(np.abs(gx), 90) + np.percentile(np.abs(gy), 90)) + 1e-6
    scale = 6.0 / p90
    nx = -gx * scale
    ny = -gy * scale
    nz = np.ones_like(gray, dtype=np.float32)
    nlen = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    shading = (nx * lx + ny * ly + nz * lz) / nlen
    shading = np.clip(shading, 0.0, None)

    p1, p99 = np.percentile(shading, (2, 98))
    shading = np.clip((shading - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
    return shading.astype(np.float32), (lx, ly, lz)


class SIRFSNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.SIRFS",
            display_name="SIRFS",
            category=CATEGORY,
            description="基于梯度的光照方向与着色图估计（轻量 SfS 近似，非官方 SIRFS）。",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("SHADING"), io.String.Output("LIGHT_DIR")],
        )

    @classmethod
    def execute(cls, image):
        outs, dirs = [], []
        for pil in tensor_to_pils(image):
            rgb = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
            shading, (lx, ly, lz) = _estimate_light_and_shading(rgb)
            outs.append(array_to_image(np.repeat(shading[..., None], 3, axis=-1)))
            dirs.append(f"{lx:.4f},{ly:.4f},{lz:.4f}")
        return io.NodeOutput(torch.cat(outs, dim=0), "\n".join(dirs))
