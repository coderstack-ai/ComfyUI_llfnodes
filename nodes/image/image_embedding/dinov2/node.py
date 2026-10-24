from comfy_api.latest import io
from ...cv_shared import get_cached_model, get_torch_device, tensor_to_pils
import torch
from ....model_store import hf_local

CATEGORY = "llfnodes/图像处理/图像嵌入"
MODELS = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
}


class DINOv2EmbedNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DINOv2",
            display_name="DINOv2",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS), default="dinov2-small"),
            ],
            outputs=[io.String.Output("EMBEDDING")],
        )

    @classmethod
    def execute(cls, image, model):
        from transformers import AutoImageProcessor, AutoModel

        mid = MODELS[model]
        local = hf_local(mid)

        def loader(device):
            return AutoImageProcessor.from_pretrained(local, local_files_only=True), AutoModel.from_pretrained(local, local_files_only=True).to(device).eval()

        processor, net = get_cached_model(f"dinov2:{mid}", loader)
        device = get_torch_device()
        lines = []
        for pil in tensor_to_pils(image):
            inputs = processor(images=pil, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = net(**inputs)
                feat = out.last_hidden_state[:, 0]
                feat = feat / feat.norm(dim=-1, keepdim=True)
            vec = feat[0].detach().cpu().tolist()
            lines.append(",".join(f"{v:.6f}" for v in vec[:64]) + f"... dim={len(vec)}")
        return io.NodeOutput("\n---\n".join(lines))
