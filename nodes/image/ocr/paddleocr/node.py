from comfy_api.latest import io
from ...cv_shared import array_to_image, get_cached_model, require_pkg, tensor_to_pils
import numpy as np
import torch
import cv2

CATEGORY = "llfnodes/图像处理/图像文字识别"


class PaddleOCRNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.PaddleOCR",
            display_name="PaddleOCR",
            category=CATEGORY,
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("IMAGE"), io.String.Output("TEXT")],
        )

    @classmethod
    def execute(cls, image):
        require_pkg("paddleocr", "pip install paddleocr paddlepaddle")
        from paddleocr import PaddleOCR

        ocr = get_cached_model("paddleocr", lambda d: PaddleOCR(use_angle_cls=True, lang="ch", show_log=False))
        outs, texts = [], []
        for pil in tensor_to_pils(image):
            arr = np.asarray(pil.convert("RGB"))
            result = ocr.ocr(arr, cls=True)
            canvas = arr.copy()
            lines = []
            if result and result[0]:
                for line in result[0]:
                    box, (txt, score) = line
                    pts = np.array(box, dtype=np.int32)
                    cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)
                    lines.append(f"{txt}:{score:.3f}")
            outs.append(array_to_image(canvas))
            texts.append("\n".join(lines))
        return io.NodeOutput(torch.cat(outs, dim=0), "\n---\n".join(texts))
