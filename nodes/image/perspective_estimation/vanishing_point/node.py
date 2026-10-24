from comfy_api.latest import io
from ...cv_shared import array_to_image, tensor_to_pils
import numpy as np
import torch
import cv2

CATEGORY = "llfnodes/图像处理/透视与消失点估计"


class VanishingPointNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.VanishingPoint",
            display_name="Vanishing Point",
            category=CATEGORY,
            description="基于 LSD 线段的消失点近似估计。",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("IMAGE"), io.String.Output("VANISH_POINT")],
        )

    @classmethod
    def execute(cls, image):
        outs, texts = [], []
        for pil in tensor_to_pils(image):
            arr = np.asarray(pil.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            lsd = cv2.createLineSegmentDetector(0)
            lines = lsd.detect(gray)[0]
            canvas = arr.copy()
            pts = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = map(int, line[0])
                    cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    pts.append(((x1 + x2) / 2, (y1 + y2) / 2))
            if pts:
                vp = np.mean(np.array(pts), axis=0)
                cv2.circle(canvas, (int(vp[0]), int(vp[1])), 8, (255, 0, 0), -1)
                texts.append(f"{vp[0]:.1f},{vp[1]:.1f}")
            else:
                texts.append("n/a")
            outs.append(array_to_image(canvas))
        return io.NodeOutput(torch.cat(outs, dim=0), "\n---\n".join(texts))
