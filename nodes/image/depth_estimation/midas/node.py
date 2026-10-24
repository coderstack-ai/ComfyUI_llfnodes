from comfy_api.latest import io

from ..common import CATEGORY, run_hf_depth

MODELS = {
    "DPT-Hybrid": "Intel/dpt-hybrid-midas",
    "DPT-Large": "Intel/dpt-large",
    "DPT-BEiT-Large": "Intel/dpt-beit-large-512",
}


class MiDaSNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.MiDaS",
            display_name="MiDaS",
            category=CATEGORY,
            description="MiDaS / DPT 相对深度估计。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="DPT-Hybrid"),
                io.Boolean.Input("invert", default=False, label_on="near=black", label_off="near=white"),
            ],
            outputs=[io.Image.Output("DEPTH")],
        )

    @classmethod
    def execute(cls, image, model, invert):
        return io.NodeOutput(run_hf_depth(MODELS[model], image, invert=invert))
