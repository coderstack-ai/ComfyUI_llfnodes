from comfy_api.latest import io

import comfy.model_management as model_management

from ..common import CATEGORY, ensure_controlnet_aux, get_cached_model, run_controlnet_pose_detector

BBOX_OPTIONS = [
    "yolox_l.onnx",
    "yolox_l.torchscript.pt",
    "yolo_nas_l_fp16.onnx",
    "yolo_nas_m_fp16.onnx",
    "yolo_nas_s_fp16.onnx",
    "None",
]
POSE_OPTIONS = [
    "dw-ll_ucoco_384_bs5.torchscript.pt",
    "dw-ll_ucoco_384.onnx",
    "dw-ll_ucoco.onnx",
]


class DWPoseNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DWPose",
            display_name="DWPose",
            category=CATEGORY,
            description="DWPose 人体姿态估计。",
            inputs=[
                io.Image.Input("image"),
                io.Boolean.Input("detect_body", default=True),
                io.Boolean.Input("detect_hand", default=True),
                io.Boolean.Input("detect_face", default=True),
                io.Combo.Input("bbox_detector", options=BBOX_OPTIONS, default="yolox_l.onnx"),
                io.Combo.Input("pose_estimator", options=POSE_OPTIONS, default="dw-ll_ucoco_384_bs5.torchscript.pt"),
                io.Int.Input("resolution", default=512, min=64, max=2048, step=64),
            ],
            outputs=[io.Image.Output("POSE")],
        )

    @classmethod
    def execute(cls, image, detect_body, detect_hand, detect_face, bbox_detector, pose_estimator, resolution):
        ensure_controlnet_aux()
        from custom_controlnet_aux.dwpose import DwposeDetector

        dwpose_repo = "yzd-v/DWPose"
        if bbox_detector == "None":
            yolo_repo = dwpose_repo
            det_filename = None
        elif bbox_detector == "yolox_l.onnx":
            yolo_repo = dwpose_repo
            det_filename = bbox_detector
        elif "yolox" in bbox_detector:
            yolo_repo = "hr16/yolox-onnx"
            det_filename = bbox_detector
        elif "yolo_nas" in bbox_detector:
            yolo_repo = "hr16/yolo-nas-fp16"
            det_filename = bbox_detector
        else:
            raise ValueError(f"Unsupported bbox detector: {bbox_detector}")

        if pose_estimator == "dw-ll_ucoco_384.onnx":
            pose_repo = dwpose_repo
        elif pose_estimator.endswith(".onnx"):
            pose_repo = "hr16/UnJIT-DWPose"
        elif pose_estimator.endswith(".torchscript.pt"):
            pose_repo = "hr16/DWPose-TorchScript-BatchSize5"
        else:
            raise ValueError(f"Unsupported pose estimator: {pose_estimator}")

        cache_key = f"dwpose:{bbox_detector}:{pose_estimator}"

        def loader(device):
            return DwposeDetector.from_pretrained(
                pose_repo,
                yolo_repo,
                det_filename=det_filename,
                pose_filename=pose_estimator,
                torchscript_device=model_management.get_torch_device(),
            )

        detector = get_cached_model(cache_key, loader)
        out = run_controlnet_pose_detector(
            detector,
            image,
            detect_resolution=resolution,
            include_body=detect_body,
            include_hand=detect_hand,
            include_face=detect_face,
            image_and_json=False,
        )
        return io.NodeOutput(out)
