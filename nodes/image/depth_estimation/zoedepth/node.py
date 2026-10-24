from comfy_api.latest import io

from ..common import CATEGORY, run_hf_depth

MODELS = {
    "NYU": "Intel/zoedepth-nyu",
    "NYU-KITTI": "Intel/zoedepth-nyu-kitti",
}


class ZoeDepthNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.ZoeDepth",
            display_name="ZoeDepth",
            category=CATEGORY,
            description="ZoeDepth 度量深度估计。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="NYU-KITTI"),
                io.Boolean.Input("invert", default=False, label_on="near=black", label_off="near=white"),
            ],
            outputs=[io.Image.Output("DEPTH")],
        )

    @classmethod
    def execute(cls, image, model, invert):
        return io.NodeOutput(run_hf_depth(MODELS[model], image, invert=invert))
