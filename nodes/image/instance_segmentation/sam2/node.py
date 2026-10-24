from comfy_api.latest import io

from ..common import CATEGORY, run_mask_generation

MODELS = {
    "Hiera-Tiny": "facebook/sam2.1-hiera-tiny",
    "Hiera-Small": "facebook/sam2.1-hiera-small",
    "Hiera-Base-Plus": "facebook/sam2.1-hiera-base-plus",
    "Hiera-Large": "facebook/sam2.1-hiera-large",
}


class SAM2Node(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.SAM2",
            display_name="SAM 2",
            category=CATEGORY,
            description="Segment Anything 2 自动掩码生成。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="Hiera-Tiny"),
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
