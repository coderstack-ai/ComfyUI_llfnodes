from comfy_api.latest import io

from ..common import CATEGORY, run_oneformer

# Hub 上不存在 oneformer_coco_swin_tiny；官方仅有下列权重。
MODELS = {
    "ADE20K-Swin-Tiny": "shi-labs/oneformer_ade20k_swin_tiny",
    "ADE20K-Swin-Large": "shi-labs/oneformer_ade20k_swin_large",
    "COCO-Swin-Large": "shi-labs/oneformer_coco_swin_large",
    "Cityscapes-Swin-Large": "shi-labs/oneformer_cityscapes_swin_large",
}


class OneFormerNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.OneFormer",
            display_name="OneFormer",
            category=CATEGORY,
            description="OneFormer 通用分割（语义模式）。",
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
        seg, label_map = run_oneformer(MODELS[model], image, alpha=overlay_alpha)
        return io.NodeOutput(seg, label_map)
