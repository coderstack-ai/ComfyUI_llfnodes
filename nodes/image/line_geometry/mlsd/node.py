from comfy_api.latest import io
from ...cv_shared import ensure_controlnet_aux, get_cached_model, run_controlnet_detector

CATEGORY = "llfnodes/图像处理/线段与几何结构检测"


class MLSDNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.MLSD",
            display_name="M-LSD",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("score_thr", default=0.1, min=0.0, max=1.0, step=0.01),
                io.Float.Input("dist_thr", default=0.1, min=0.0, max=20.0, step=0.01),
                io.Int.Input("resolution", default=512, min=64, max=2048, step=64),
            ],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image, score_thr, dist_thr, resolution):
        ensure_controlnet_aux()
        from custom_controlnet_aux.mlsd import MLSDdetector

        det = get_cached_model("mlsd", lambda d: MLSDdetector.from_pretrained().to(d))
        return io.NodeOutput(
            run_controlnet_detector(
                det, image, detect_resolution=resolution, thr_v=score_thr, thr_d=dist_thr
            )
        )
