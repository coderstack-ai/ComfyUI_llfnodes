from comfy_api.latest import io
from ...cv_shared import array_to_image, tensor_to_pils
import numpy as np
import torch
import cv2

CATEGORY = "llfnodes/图像处理/图像文字识别"


class DBNetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DBNet",
            display_name="DBNet",
            category=CATEGORY,
            description="DBNet 风格文本区域检测（OpenCV MSER 近似实现）。",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image):
        outs = []
        for pil in tensor_to_pils(image):
            arr = np.asarray(pil.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            canvas = arr.copy()
            for p in regions:
                x, y, w, h = cv2.boundingRect(p)
                if w * h < 80:
                    continue
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
            outs.append(array_to_image(canvas))
        return io.NodeOutput(torch.cat(outs, dim=0))
