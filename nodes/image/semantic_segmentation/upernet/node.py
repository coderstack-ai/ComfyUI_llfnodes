from comfy_api.latest import io

from ..common import CATEGORY, run_semantic_auto_model

MODELS = {
    "Swin-Tiny": "openmmlab/upernet-swin-tiny",
    "Swin-Small": "openmmlab/upernet-swin-small",
    "Swin-Base": "openmmlab/upernet-swin-base",
    "Swin-Large": "openmmlab/upernet-swin-large",
    "ConvNeXt-Tiny": "openmmlab/upernet-convnext-tiny",
    "ConvNeXt-Small": "openmmlab/upernet-convnext-small",
    "ConvNeXt-Base": "openmmlab/upernet-convnext-base",
    "ConvNeXt-Large": "openmmlab/upernet-convnext-large",
}


class UPerNetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.UPerNet",
            display_name="UPerNet",
            category=CATEGORY,
            description="UPerNet 语义分割（Swin / ConvNeXt backbone）。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="Swin-Tiny"),
                io.Float.Input("overlay_alpha", default=0.6, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[
                io.Image.Output("SEGMENTATION"),
                io.Image.Output("LABEL_MAP"),
            ],
        )

    @classmethod
    def execute(cls, image, model, overlay_alpha):
        seg, label_map = run_semantic_auto_model(MODELS[model], image, alpha=overlay_alpha)
        return io.NodeOutput(seg, label_map)
