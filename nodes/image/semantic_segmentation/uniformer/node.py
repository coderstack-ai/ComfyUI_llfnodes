import torch
from comfy_api.latest import io

from ....model_store import hf_local
from ..common import CATEGORY, colorize_and_blend, get_cached_model, get_torch_device, label_map_to_mask_image, tensor_to_pils
from .model import REPO_ID, load_uniformer_ade20k, predict_label_map

MODELS = {
    "S-ADE20K": REPO_ID,
}


class UniFormerNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.UniFormer",
            display_name="UniFormer",
            category=CATEGORY,
            description="UniFormer-S + UPerNet ADE20K 语义分割（权重：models/llfnodes_hf/）。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="S-ADE20K"),
                io.Float.Input("overlay_alpha", default=0.6, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[
                io.Image.Output("SEGMENTATION"),
                io.Image.Output("LABEL_MAP"),
            ],
        )

    @classmethod
    def execute(cls, image, model, overlay_alpha):
        repo_id = MODELS[model]
        local = hf_local(repo_id)

        def loader(device):
            return load_uniformer_ade20k(local, device)

        net = get_cached_model(f"uniformer:{local}", loader)
        device = get_torch_device()
        overlays = []
        maps = []
        for pil in tensor_to_pils(image):
            label_map = predict_label_map(net, pil, device)
            overlays.append(colorize_and_blend(pil, label_map, overlay_alpha))
            maps.append(label_map_to_mask_image(label_map))
        return io.NodeOutput(torch.cat(overlays, dim=0), torch.cat(maps, dim=0))
