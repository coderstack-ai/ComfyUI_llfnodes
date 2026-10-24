from comfy_api.latest import io

from ..common import (
    CATEGORY,
    ensure_controlnet_aux,
    get_cached_model,
    run_controlnet_normal_detector,
)


class NormalBaeNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.NormalBae",
            display_name="NormalBae",
            category=CATEGORY,
            description="NormalBae 表面法线估计（依赖 comfyui_controlnet_aux）。",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("resolution", default=512, min=64, max=2048, step=64),
            ],
            outputs=[io.Image.Output("NORMAL")],
        )

    @classmethod
    def execute(cls, image, resolution):
        ensure_controlnet_aux()
        from custom_controlnet_aux.normalbae import NormalBaeDetector

        def loader(device):
            return NormalBaeDetector.from_pretrained().to(device)

        detector = get_cached_model("normalbae", loader)
        out = run_controlnet_normal_detector(
            detector,
            image,
            detect_resolution=resolution,
        )
        return io.NodeOutput(out)
