from comfy_api.latest import io

from ..common import CATEGORY, ensure_controlnet_aux, get_cached_model, run_controlnet_detector


class PiDiNetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.PiDiNet",
            display_name="PiDiNet",
            category=CATEGORY,
            description="PiDiNet 软边缘线稿提取。",
            inputs=[
                io.Image.Input("image"),
                io.Boolean.Input("safe", default=True),
                io.Int.Input("resolution", default=512, min=64, max=2048, step=64),
            ],
            outputs=[io.Image.Output("EDGE")],
        )

    @classmethod
    def execute(cls, image, safe, resolution):
        ensure_controlnet_aux()
        from custom_controlnet_aux.pidi import PidiNetDetector

        def loader(device):
            return PidiNetDetector.from_pretrained().to(device)

        detector = get_cached_model("pidinet", loader)
        out = run_controlnet_detector(
            detector,
            image,
            detect_resolution=resolution,
            safe=safe,
        )
        return io.NodeOutput(out)
