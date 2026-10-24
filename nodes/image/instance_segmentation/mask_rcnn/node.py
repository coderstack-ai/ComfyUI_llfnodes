from comfy_api.latest import io

from ..common import CATEGORY, run_mask_rcnn


class MaskRCNNNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.MaskRCNN",
            display_name="Mask R-CNN",
            category=CATEGORY,
            description="Mask R-CNN 实例分割（torchvision ResNet50-FPN-V2）。",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("score_threshold", default=0.5, min=0.0, max=1.0, step=0.01),
                io.Float.Input("overlay_alpha", default=0.6, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[
                io.Image.Output("SEGMENTATION"),
                io.Mask.Output("MASK"),
            ],
        )

    @classmethod
    def execute(cls, image, score_threshold, overlay_alpha):
        seg, mask = run_mask_rcnn(image, score_threshold=score_threshold, alpha=overlay_alpha)
        return io.NodeOutput(seg, mask)
