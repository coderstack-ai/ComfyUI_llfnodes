from comfy_api.latest import io
from ...cv_shared import get_cached_model, get_torch_device, overlay_mask, pils_to_tensor, tensor_to_pils
from ....model_store import hf_local

CATEGORY = "llfnodes/图像处理/全景分割"
# Hub 上不存在 oneformer_coco_swin_tiny。
MODELS = {
    "COCO-Swin-Large": "shi-labs/oneformer_coco_swin_large",
    "ADE20K-Swin-Tiny": "shi-labs/oneformer_ade20k_swin_tiny",
    "ADE20K-Swin-Large": "shi-labs/oneformer_ade20k_swin_large",
}


class PanopticOneFormerNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.PanopticOneFormer",
            display_name="OneFormer",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS), default="ADE20K-Swin-Tiny"),
                io.Float.Input("overlay_alpha", default=0.55, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image, model, overlay_alpha):
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        mid = MODELS[model]
        local = hf_local(mid)

        def loader(device):
            return (
                OneFormerProcessor.from_pretrained(local, local_files_only=True),
                OneFormerForUniversalSegmentation.from_pretrained(local, local_files_only=True).to(device).eval(),
            )

        processor, net = get_cached_model(f"panoptic-oneformer:{mid}", loader)
        device = get_torch_device()
        outs = []
        for pil in tensor_to_pils(image):
            inputs = processor(images=pil, task_inputs=["panoptic"], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = net(**inputs)
            result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[pil.size[::-1]])[0]
            seg = result["segmentation"].detach().cpu().numpy()
            outs.append(overlay_mask(pil, seg, alpha=overlay_alpha))
        return io.NodeOutput(pils_to_tensor(outs))
