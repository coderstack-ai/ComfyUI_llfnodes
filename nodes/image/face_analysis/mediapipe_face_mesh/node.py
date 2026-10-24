from pathlib import Path

import cv2
import numpy as np
import torch
from comfy_api.latest import io

from ...cv_shared import array_to_image, require_pkg, tensor_to_pils

CATEGORY = "llfnodes/图像处理/人脸分析"


class MediaPipeFaceMeshNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.MediaPipeFaceMesh",
            display_name="MediaPipe Face Mesh",
            category=CATEGORY,
            description="MediaPipe Face Mesh；缺 mediapipe 或 face_landmarker.task 时直接报错。",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image):
        require_pkg("mediapipe", "pip install mediapipe")
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(__file__).resolve().parents[4] / "models" / "face_landmarker.task"
        if not model_path.exists():
            raise RuntimeError(
                f"缺少 Face Landmarker 模型: {model_path}。"
                "请从 MediaPipe 官方下载 face_landmarker.task 放到 ComfyUI/models/"
            )

        base = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)
        outs = []
        for pil in tensor_to_pils(image):
            arr = np.asarray(pil.convert("RGB")).copy()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
            result = detector.detect(mp_image)
            h, w = arr.shape[:2]
            for face in result.face_landmarks:
                for lm in face:
                    cv2.circle(arr, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)
            outs.append(array_to_image(arr))
        return io.NodeOutput(torch.cat(outs, dim=0))
