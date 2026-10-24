from pathlib import Path

import cv2
import numpy as np
import torch
from comfy_api.latest import io

from ...cv_shared import array_to_image, tensor_to_pils

CATEGORY = "llfnodes/图像处理/人脸分析"


def _opencv_face_fallback(image):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    outs = []
    for pil in tensor_to_pils(image):
        arr = np.asarray(pil.convert("RGB")).copy()
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4)
        for x, y, w, h in faces:
            cv2.rectangle(arr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        outs.append(array_to_image(arr))
    return io.NodeOutput(torch.cat(outs, dim=0))


class MediaPipeFaceMeshNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.MediaPipeFaceMesh",
            display_name="MediaPipe Face Mesh",
            category=CATEGORY,
            description="MediaPipe Face Mesh；缺 mediapipe/权重时回退 OpenCV Haar 人脸框。",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError:
            return _opencv_face_fallback(image)

        model_path = Path(__file__).resolve().parents[4] / "models" / "face_landmarker.task"
        if not model_path.exists():
            return _opencv_face_fallback(image)

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
