import cv2
import numpy as np
import torch
from comfy_api.latest import io

from ..common import CATEGORY, array_to_image, device_str_for_onnx, get_cached_model, tensor_to_pils

MODES = ["balanced", "performance", "lightweight"]


def _opencv_pose_fallback(image):
    """HOG person boxes when rtmlib is unavailable — keeps the workflow runnable."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    outs = []
    for pil in tensor_to_pils(image):
        bgr = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        canvas = np.zeros_like(bgr)
        rects, _ = hog.detectMultiScale(bgr, winStride=(8, 8), padding=(8, 8), scale=1.05)
        for x, y, w, h in rects:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # crude stick figure
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(canvas, (cx, y + h // 6), max(3, w // 12), (0, 255, 0), -1)
            cv2.line(canvas, (cx, y + h // 4), (cx, y + 3 * h // 4), (0, 255, 0), 2)
            cv2.line(canvas, (cx, y + h // 3), (x + w // 6, cy), (0, 255, 0), 2)
            cv2.line(canvas, (cx, y + h // 3), (x + 5 * w // 6, cy), (0, 255, 0), 2)
            cv2.line(canvas, (cx, y + 3 * h // 4), (x + w // 4, y + h - 2), (0, 255, 0), 2)
            cv2.line(canvas, (cx, y + 3 * h // 4), (x + 3 * w // 4, y + h - 2), (0, 255, 0), 2)
        outs.append(array_to_image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
    return io.NodeOutput(torch.cat(outs, dim=0))


class RTMPoseNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.RTMPose",
            display_name="RTMPose",
            category=CATEGORY,
            description="RTMPose（需 rtmlib）；未安装时回退 OpenCV HOG 人体框+简笔画骨架。",
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
        except ImportError:
            return _opencv_pose_fallback(image)

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
