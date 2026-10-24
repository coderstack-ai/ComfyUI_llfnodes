from comfy_api.latest import io

from ..common import CATEGORY, run_mask2former

MODELS = {
    "ADE20K-Swin-Tiny": "facebook/mask2former-swin-tiny-ade-semantic",
    "ADE20K-Swin-Small": "facebook/mask2former-swin-small-ade-semantic",
    "ADE20K-Swin-Base": "facebook/mask2former-swin-base-ade-semantic",
    "ADE20K-Swin-Large": "facebook/mask2former-swin-large-ade-semantic",
    "Cityscapes-Swin-Large": "facebook/mask2former-swin-large-cityscapes-semantic",
}


class Mask2FormerNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.Mask2Former",
            display_name="Mask2Former",
            category=CATEGORY,
            description="Mask2Former 语义分割。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="ADE20K-Swin-Tiny"),
                io.Float.Input("overlay_alpha", default=0.6, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[
                io.Image.Output("SEGMENTATION"),
                io.Image.Output("LABEL_MAP"),
            ],
        )

    @classmethod
    def execute(cls, image, model, overlay_alpha):
        seg, label_map = run_mask2former(MODELS[model], image, alpha=overlay_alpha)
        return io.NodeOutput(seg, label_map)
