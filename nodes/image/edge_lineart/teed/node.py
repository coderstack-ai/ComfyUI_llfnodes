from comfy_api.latest import io

from ..common import CATEGORY, ensure_controlnet_aux, get_cached_model, run_controlnet_detector


class TEEDNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.TEED",
            display_name="TEED",
            category=CATEGORY,
            description="TEED 软边缘线稿提取。",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("safe_steps", default=2, min=0, max=10, step=1),
                io.Int.Input("resolution", default=512, min=64, max=2048, step=64),
            ],
            outputs=[io.Image.Output("EDGE")],
        )

    @classmethod
    def execute(cls, image, safe_steps, resolution):
        ensure_controlnet_aux()
        from custom_controlnet_aux.teed import TEDDetector

        def loader(device):
            return TEDDetector.from_pretrained().to(device)

        detector = get_cached_model("teed", loader)
        out = run_controlnet_detector(
            detector,
            image,
            detect_resolution=resolution,
            safe_steps=safe_steps,
        )
        return io.NodeOutput(out)
