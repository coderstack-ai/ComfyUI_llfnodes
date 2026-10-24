from comfy_api.latest import io

from ..common import CATEGORY, run_semantic_auto_model

MODELS = {
    "B0-ADE20K": "nvidia/segformer-b0-finetuned-ade-512-512",
    "B1-ADE20K": "nvidia/segformer-b1-finetuned-ade-512-512",
    "B2-ADE20K": "nvidia/segformer-b2-finetuned-ade-512-512",
    "B3-ADE20K": "nvidia/segformer-b3-finetuned-ade-512-512",
    "B4-ADE20K": "nvidia/segformer-b4-finetuned-ade-512-512",
    "B5-ADE20K": "nvidia/segformer-b5-finetuned-ade-640-640",
    "B1-Cityscapes": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
}


class SegFormerNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.SegFormer",
            display_name="SegFormer",
            category=CATEGORY,
            description="SegFormer 语义分割（HuggingFace Transformers）。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="B0-ADE20K"),
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
