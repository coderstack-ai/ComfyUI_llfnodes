import cv2
import numpy as np
import torch
from comfy_api.latest import io

from ...cv_shared import array_to_image, tensor_to_pils

CATEGORY = "llfnodes/图像处理/图像修复"


class MATNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.MAT",
            display_name="MAT",
            category=CATEGORY,
            description="MAT 官方权重未内置；当前用 OpenCV Telea 修复作为可运行回退。",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.String.Input("model_path", default="", tooltip="可选官方权重（当前回退不使用）"),
            ],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image, mask, model_path=""):
        masks = mask.unsqueeze(0) if mask.ndim == 2 else mask
        outs = []
        for i, pil in enumerate(tensor_to_pils(image)):
            bgr = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
            m = masks[min(i, masks.shape[0] - 1)].detach().cpu().numpy()
            m = (m * 255.0).clip(0, 255).astype(np.uint8)
            if m.ndim == 3:
                m = m.squeeze()
            if m.shape[:2] != bgr.shape[:2]:
                m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            fixed = cv2.inpaint(bgr, m, 3, cv2.INPAINT_TELEA)
            outs.append(array_to_image(cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB)))
        return io.NodeOutput(torch.cat(outs, dim=0))
