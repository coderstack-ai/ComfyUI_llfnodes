import cv2
import numpy as np
import torch
from comfy_api.latest import io

from ...cv_shared import array_to_image, tensor_to_pils

CATEGORY = "llfnodes/图像处理/材质与固有图像分解"


class IntrinsicImageNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.IntrinsicImage",
            display_name="Intrinsic Image",
            category=CATEGORY,
            description="Retinex 固有图像分解：大尺度亮度作 shading，rgb/shading 作 albedo。",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("ALBEDO"), io.Image.Output("SHADING")],
        )

    @classmethod
    def execute(cls, image):
        outs_a, outs_s = [], []
        for pil in tensor_to_pils(image):
            rgb = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
            h, w = rgb.shape[:2]
            lum = rgb.mean(axis=-1).astype(np.float32)
            sigma = max(4.0, min(h, w) / 24.0)
            shading = cv2.GaussianBlur(lum, (0, 0), sigmaX=sigma)
            shading = np.clip(shading, 1e-3, 1.0)
            albedo = np.clip(rgb / shading[..., None], 0.0, 1.0)
            a_max = float(np.percentile(albedo, 99))
            if a_max > 1.0:
                albedo = np.clip(albedo / a_max, 0.0, 1.0)
            shading3 = np.repeat(shading[..., None], 3, axis=-1)
            outs_a.append(array_to_image(albedo))
            outs_s.append(array_to_image(shading3))
        return io.NodeOutput(torch.cat(outs_a, dim=0), torch.cat(outs_s, dim=0))
