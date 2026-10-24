from comfy_api.latest import io
from ...cv_shared import get_cached_model, get_torch_device, tensor_to_pils
import torch
from ....model_store import hf_local

CATEGORY = "llfnodes/图像处理/图像嵌入"
MODELS = {
    "siglip-base": "google/siglip-base-patch16-224",
    "siglip-so400m": "google/siglip-so400m-patch14-384",
}


class SigLIPEmbedNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.SigLIP",
            display_name="SigLIP",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS), default="siglip-base"),
            ],
            outputs=[io.String.Output("EMBEDDING")],
        )

    @classmethod
    def execute(cls, image, model):
        from transformers import AutoModel, AutoProcessor

        mid = MODELS[model]
        local = hf_local(mid)

        def loader(device):
            return AutoProcessor.from_pretrained(local, local_files_only=True), AutoModel.from_pretrained(local, local_files_only=True).to(device).eval()

        processor, net = get_cached_model(f"siglip:{mid}", loader)
        device = get_torch_device()
        lines = []
        for pil in tensor_to_pils(image):
            inputs = processor(images=pil, return_tensors="pt").to(device)
            with torch.inference_mode():
                feat = net.get_image_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            vec = feat[0].detach().cpu().tolist()
            lines.append(",".join(f"{v:.6f}" for v in vec[:64]) + f"... dim={len(vec)}")
        return io.NodeOutput("\n---\n".join(lines))
