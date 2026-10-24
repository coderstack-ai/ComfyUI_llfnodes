from comfy_api.latest import io
from ...cv_shared import get_cached_model, get_torch_device, overlay_mask, pils_to_tensor, tensor_to_pils
from ....model_store import hf_local

CATEGORY = "llfnodes/图像处理/全景分割"
MODELS = {
    "COCO-Swin-Tiny": "facebook/mask2former-swin-tiny-coco-panoptic",
    "COCO-Swin-Small": "facebook/mask2former-swin-small-coco-panoptic",
    "COCO-Swin-Large": "facebook/mask2former-swin-large-coco-panoptic",
}


class PanopticMask2FormerNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.PanopticMask2Former",
            display_name="Mask2Former",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS), default="COCO-Swin-Tiny"),
                io.Float.Input("overlay_alpha", default=0.55, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image, model, overlay_alpha):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        mid = MODELS[model]
        local = hf_local(mid)

        def loader(device):
            return (
                AutoImageProcessor.from_pretrained(local, local_files_only=True),
                Mask2FormerForUniversalSegmentation.from_pretrained(local, local_files_only=True).to(device).eval(),
            )

        processor, net = get_cached_model(f"panoptic-m2f:{mid}", loader)
        device = get_torch_device()
        outs = []
        for pil in tensor_to_pils(image):
            inputs = {k: v.to(device) for k, v in processor(images=pil, return_tensors="pt").items()}
            outputs = net(**inputs)
            result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[pil.size[::-1]])[0]
            seg = result["segmentation"].detach().cpu().numpy()
            outs.append(overlay_mask(pil, seg, alpha=overlay_alpha))
        return io.NodeOutput(pils_to_tensor(outs))
