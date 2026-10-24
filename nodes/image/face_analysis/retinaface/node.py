from comfy_api.latest import io
from ...cv_shared import array_to_image, get_cached_model, require_pkg, tensor_to_pils
import numpy as np
import torch
import cv2

CATEGORY = "llfnodes/图像处理/人脸分析"


class RetinaFaceNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.RetinaFace",
            display_name="RetinaFace",
            category=CATEGORY,
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("IMAGE"), io.String.Output("FACES")],
        )

    @classmethod
    def execute(cls, image):
        require_pkg("insightface", "pip install insightface onnxruntime")
        from insightface.app import FaceAnalysis

        def loader(device):
            app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            app.prepare(ctx_id=0 if str(device).startswith("cuda") else -1, det_size=(640, 640))
            return app

        app = get_cached_model("retinaface-buffalo", loader)
        outs, texts = [], []
        for pil in tensor_to_pils(image):
            arr = np.asarray(pil.convert("RGB"))
            faces = app.get(arr[:, :, ::-1])
            canvas = arr.copy()
            lines = []
            for f in faces:
                box = f.bbox.astype(int)
                cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                lines.append(f"{box.tolist()} score={float(f.det_score):.3f}")
            outs.append(array_to_image(canvas))
            texts.append("\n".join(lines))
        return io.NodeOutput(torch.cat(outs, dim=0), "\n---\n".join(texts))
