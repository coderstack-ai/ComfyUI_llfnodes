from comfy_api.latest import io

from ..common import CATEGORY, run_mask_generation

MODELS = {
    "ViT-Base": "facebook/sam-vit-base",
    "ViT-Large": "facebook/sam-vit-large",
    "ViT-Huge": "facebook/sam-vit-huge",
}


class SAMNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.SAM",
            display_name="SAM",
            category=CATEGORY,
            description="Segment Anything (SAM) 自动掩码生成。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="ViT-Base"),
                io.Int.Input("points_per_batch", default=64, min=1, max=256, step=1),
                io.Float.Input("overlay_alpha", default=0.6, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[
                io.Image.Output("SEGMENTATION"),
                io.Mask.Output("MASK"),
            ],
        )

    @classmethod
    def execute(cls, image, model, points_per_batch, overlay_alpha):
        seg, mask = run_mask_generation(
            MODELS[model], image, points_per_batch=points_per_batch, alpha=overlay_alpha
        )
        return io.NodeOutput(seg, mask)
