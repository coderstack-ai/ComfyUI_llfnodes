from comfy_api.latest import io

from ..common import CATEGORY, run_hf_depth

MODELS = {
    "Small": "depth-anything/Depth-Anything-V2-Small-hf",
    "Base": "depth-anything/Depth-Anything-V2-Base-hf",
    "Large": "depth-anything/Depth-Anything-V2-Large-hf",
}


class DepthAnythingV2Node(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DepthAnythingV2",
            display_name="Depth Anything V2",
            category=CATEGORY,
            description="Depth Anything V2 相对深度估计（HuggingFace Transformers）。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="Small"),
                io.Boolean.Input("invert", default=False, label_on="near=black", label_off="near=white"),
            ],
            outputs=[io.Image.Output("DEPTH")],
        )

    @classmethod
    def execute(cls, image, model, invert):
        return io.NodeOutput(run_hf_depth(MODELS[model], image, invert=invert))
