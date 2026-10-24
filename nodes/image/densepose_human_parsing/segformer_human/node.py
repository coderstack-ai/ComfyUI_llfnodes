from comfy_api.latest import io
from ...cv_shared import array_to_image, get_cached_model, get_torch_device, tensor_to_pils
import numpy as np
import torch
import torch.nn.functional as F
from ....model_store import hf_local

CATEGORY = "llfnodes/图像处理/人体密集姿态与人体解析"
MODELS = {"segformer-b2-clothes": "mattmdjaga/segformer_b2_clothes"}


class SegFormerHumanNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.SegFormerHuman",
            display_name="SegFormer Human Parsing",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS), default="segformer-b2-clothes"),
            ],
            outputs=[io.Image.Output("IMAGE"), io.Mask.Output("MASK")],
        )

    @classmethod
    def execute(cls, image, model):
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        mid = MODELS[model]
        local = hf_local(mid)

        def loader(device):
            return (
                AutoImageProcessor.from_pretrained(local, local_files_only=True),
                SegformerForSemanticSegmentation.from_pretrained(local, local_files_only=True).to(device).eval(),
            )

        processor, net = get_cached_model(f"segformer-human:{mid}", loader)
        device = get_torch_device()
        outs, masks = [], []
        for pil in tensor_to_pils(image):
            inputs = {k: v.to(device) for k, v in processor(images=pil, return_tensors="pt").items()}
            with torch.inference_mode():
                logits = net(**inputs).logits
            up = F.interpolate(logits, size=pil.size[::-1], mode="bilinear", align_corners=False)
            pred = up.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            color = np.stack([pred * 37 % 255, pred * 17 % 255, pred * 97 % 255], axis=-1).astype(np.uint8)
            person = (pred > 0).astype(np.float32)
            outs.append(array_to_image(color))
            masks.append(torch.from_numpy(person)[None, ...])
        return io.NodeOutput(torch.cat(outs, dim=0), torch.cat(masks, dim=0))
