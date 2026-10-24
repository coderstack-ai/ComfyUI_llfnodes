import numpy as np
import torch
from comfy_api.latest import io

from ..common import (
    CATEGORY,
    array_to_image,
    draw_mediapipe_pose,
    get_cached_model,
    mediapipe_model_path,
    tensor_to_pils,
)

VARIANTS = ["lite", "full", "heavy"]


class MediaPipePoseNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.MediaPipePose",
            display_name="MediaPipe Pose",
            category=CATEGORY,
            description="MediaPipe Pose 人体姿态估计。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=VARIANTS, default="lite"),
                io.Int.Input("num_poses", default=1, min=1, max=10, step=1),
                io.Float.Input("min_confidence", default=0.5, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output("POSE")],
        )

    @classmethod
    def execute(cls, image, model, num_poses, min_confidence):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
        import mediapipe as mp

        cache_key = f"mediapipe-pose:{model}:{num_poses}"

        def loader(_device):
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=mediapipe_model_path(model)),
                running_mode=RunningMode.IMAGE,
                num_poses=num_poses,
                min_pose_detection_confidence=min_confidence,
                min_pose_presence_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )
            return PoseLandmarker.create_from_options(options)

        landmarker = get_cached_model(cache_key, loader)

        outs = []
        for pil in tensor_to_pils(image):
            rgb = np.asarray(pil)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            landmarks_list = result.pose_landmarks or []
            canvas = draw_mediapipe_pose(rgb, landmarks_list, min_score=min_confidence)
            outs.append(array_to_image(canvas))
        return io.NodeOutput(torch.cat(outs, dim=0))
