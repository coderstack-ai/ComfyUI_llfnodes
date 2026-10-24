from comfy_api.latest import io

from ..common import CATEGORY, ensure_controlnet_aux, get_cached_model, run_controlnet_pose_detector


class OpenPoseNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.OpenPose",
            display_name="OpenPose",
            category=CATEGORY,
            description="OpenPose 人体姿态估计。",
            inputs=[
                io.Image.Input("image"),
                io.Boolean.Input("detect_body", default=True),
                io.Boolean.Input("detect_hand", default=True),
                io.Boolean.Input("detect_face", default=True),
                io.Int.Input("resolution", default=512, min=64, max=2048, step=64),
            ],
            outputs=[io.Image.Output("POSE")],
        )

    @classmethod
    def execute(cls, image, detect_body, detect_hand, detect_face, resolution):
        ensure_controlnet_aux()
        from custom_controlnet_aux.open_pose import OpenposeDetector

        def loader(device):
            return OpenposeDetector.from_pretrained().to(device)

        detector = get_cached_model("openpose", loader)
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
