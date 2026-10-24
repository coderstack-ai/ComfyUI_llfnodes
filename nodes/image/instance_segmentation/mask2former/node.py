from comfy_api.latest import io

from ..common import CATEGORY, run_mask2former_instance

MODELS = {
    "COCO-Swin-Tiny": "facebook/mask2former-swin-tiny-coco-instance",
    "COCO-Swin-Small": "facebook/mask2former-swin-small-coco-instance",
    "COCO-Swin-Base": "facebook/mask2former-swin-base-coco-instance",
    "COCO-Swin-Large": "facebook/mask2former-swin-large-coco-instance",
}


class Mask2FormerInstanceNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.Mask2FormerInstance",
            display_name="Mask2Former",
            category=CATEGORY,
            description="Mask2Former 实例分割（COCO）。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="COCO-Swin-Tiny"),
                io.Float.Input("threshold", default=0.5, min=0.0, max=1.0, step=0.01),
                io.Float.Input("overlay_alpha", default=0.6, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[
                io.Image.Output("SEGMENTATION"),
                io.Mask.Output("MASK"),
            ],
        )

    @classmethod
    def execute(cls, image, model, threshold, overlay_alpha):
        seg, mask = run_mask2former_instance(
            MODELS[model], image, threshold=threshold, alpha=overlay_alpha
        )
        return io.NodeOutput(seg, mask)
