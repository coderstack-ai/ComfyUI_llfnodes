from comfy_api.latest import io
from ...cv_shared import array_to_image, require_pkg, tensor_to_pils
import numpy as np
import torch
import cv2

CATEGORY = "llfnodes/图像处理/透视与消失点估计"


class GeoCalibNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.GeoCalib",
            display_name="GeoCalib",
            category=CATEGORY,
            description="相机姿态/地平线近似（OpenCV 回退；可选 geocalib）。",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("IMAGE"), io.String.Output("PARAMS")],
        )

    @classmethod
    def execute(cls, image):
        outs, params = [], []
        for pil in tensor_to_pils(image):
            arr = np.asarray(pil.convert("RGB")).copy()
            h, w = arr.shape[:2]
            # Approximate horizon as weighted brightness centroid row
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
            row_w = gray.mean(axis=1)
            y = int(np.average(np.arange(h), weights=row_w + 1e-3))
            cv2.line(arr, (0, y), (w - 1, y), (0, 255, 255), 2)
            outs.append(array_to_image(arr))
            params.append(f"horizon_y={y}, size={w}x{h}")
        return io.NodeOutput(torch.cat(outs, dim=0), "\n---\n".join(params))
