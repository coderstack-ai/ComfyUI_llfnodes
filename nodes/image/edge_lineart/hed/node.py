from comfy_api.latest import io

from ..common import CATEGORY, ensure_controlnet_aux, get_cached_model, run_controlnet_detector


class HEDNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.HED",
            display_name="HED",
            category=CATEGORY,
            description="HED 软边缘线稿提取。",
            inputs=[
                io.Image.Input("image"),
                io.Boolean.Input("safe", default=True),
                io.Boolean.Input("scribble", default=False),
                io.Int.Input("resolution", default=512, min=64, max=2048, step=64),
            ],
            outputs=[io.Image.Output("EDGE")],
        )

    @classmethod
    def execute(cls, image, safe, scribble, resolution):
        ensure_controlnet_aux()
        from custom_controlnet_aux.hed import HEDdetector

        def loader(device):
            return HEDdetector.from_pretrained().to(device)

        detector = get_cached_model("hed", loader)
        out = run_controlnet_detector(
            detector,
            image,
            detect_resolution=resolution,
            safe=safe,
            scribble=scribble,
        )
        return io.NodeOutput(out)
