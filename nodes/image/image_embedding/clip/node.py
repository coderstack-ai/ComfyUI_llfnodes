from comfy_api.latest import io
from ...cv_shared import get_cached_model, get_torch_device, tensor_to_pils
import torch
from ....model_store import hf_local

CATEGORY = "llfnodes/图像处理/图像嵌入"
MODELS = {
    "openai-clip-vit-b32": "openai/clip-vit-base-patch32",
    "openai-clip-vit-l14": "openai/clip-vit-large-patch14",
}


class CLIPEmbedNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.CLIPEmbed",
            display_name="CLIP",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS), default="openai-clip-vit-b32"),
            ],
            outputs=[io.String.Output("EMBEDDING")],
        )

    @classmethod
    def execute(cls, image, model):
        from transformers import CLIPModel, CLIPProcessor

        mid = MODELS[model]
        local = hf_local(mid)

        def loader(device):
            return CLIPProcessor.from_pretrained(local, local_files_only=True), CLIPModel.from_pretrained(local, local_files_only=True).to(device).eval()

        processor, net = get_cached_model(f"clip:{mid}", loader)
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
