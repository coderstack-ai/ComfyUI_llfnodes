import cv2
import numpy as np
import torch
from comfy_api.latest import io

from ..common import CATEGORY, array_to_image, device_str_for_onnx, get_cached_model, tensor_to_pils

MODES = ["balanced", "performance", "lightweight"]


class RTMPoseNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.RTMPose",
            display_name="RTMPose",
            category=CATEGORY,
            description="RTMPose 人体姿态（需 rtmlib）；未安装时直接报错。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("mode", options=MODES, default="balanced"),
                io.Boolean.Input("openpose_skeleton", default=False),
                io.Float.Input("kpt_thr", default=0.5, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output("POSE")],
        )

    @classmethod
    def execute(cls, image, mode, openpose_skeleton, kpt_thr):
        try:
            from rtmlib import Body, draw_skeleton
        except ImportError as e:
            raise RuntimeError("RTMPose 需要 rtmlib。\n安装：pip install rtmlib") from e

        device = device_str_for_onnx()
        cache_key = f"rtmpose:{mode}:{openpose_skeleton}:{device}"

        def loader(_device):
            return Body(
                mode=mode,
                to_openpose=openpose_skeleton,
                backend="onnxruntime",
                device=device,
            )

        body = get_cached_model(cache_key, loader)

        outs = []
        for pil in tensor_to_pils(image):
            bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
            keypoints, scores = body(bgr)
            canvas = np.zeros_like(bgr)
            canvas = draw_skeleton(
                canvas,
                keypoints,
                scores,
                openpose_skeleton=openpose_skeleton,
                kpt_thr=kpt_thr,
            )
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            outs.append(array_to_image(rgb))
        return io.NodeOutput(torch.cat(outs, dim=0))
