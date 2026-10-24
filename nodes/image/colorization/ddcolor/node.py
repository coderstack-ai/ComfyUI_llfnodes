from comfy_api.latest import io
from ...cv_shared import array_to_image, tensor_to_pils
import numpy as np
import torch
import cv2

CATEGORY = "llfnodes/图像处理/黑白图像上色"


class DDColorNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DDColor",
            display_name="DDColor",
            category=CATEGORY,
            description="DDColor 上色（当前为 LAB 色调近似回退；可后续替换为官方权重）。",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image):
        outs = []
        for pil in tensor_to_pils(image):
            gray = np.asarray(pil.convert("L"))
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            lab[..., 1] = 128
            lab[..., 2] = 140
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            outs.append(array_to_image(rgb))
        return io.NodeOutput(torch.cat(outs, dim=0))
