from comfy_api.latest import io

from ..common import (
    CATEGORY,
    ensure_controlnet_aux,
    get_cached_model,
    run_controlnet_normal_detector,
)


class DSINENode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DSINE",
            display_name="DSINE",
            category=CATEGORY,
            description="DSINE 表面法线估计（依赖 comfyui_controlnet_aux）。",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("fov", default=60.0, min=1.0, max=180.0, step=0.5),
                io.Int.Input("iterations", default=5, min=1, max=20, step=1),
                io.Int.Input("resolution", default=512, min=64, max=2048, step=64),
            ],
            outputs=[io.Image.Output("NORMAL")],
        )

    @classmethod
    def execute(cls, image, fov, iterations, resolution):
        ensure_controlnet_aux()
        from custom_controlnet_aux.dsine import DsineDetector

        def loader(device):
            model = DsineDetector.from_pretrained().to(device)
            return model

        detector = get_cached_model("dsine", loader)
        out = run_controlnet_normal_detector(
            detector,
            image,
            fov=fov,
            iterations=iterations,
            detect_resolution=resolution,
        )
        return io.NodeOutput(out)
