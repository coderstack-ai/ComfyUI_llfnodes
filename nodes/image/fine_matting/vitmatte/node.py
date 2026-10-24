from comfy_api.latest import io
from ...cv_shared import get_cached_model, get_torch_device, tensor_to_pils, array_to_image
import numpy as np
import torch
from PIL import Image
from ....model_store import hf_local

CATEGORY = "llfnodes/图像处理/精细透明度估计"
MODELS = {"vitmatte-base": "hustvl/vitmatte-base"}


class ViTMatteNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.ViTMatte",
            display_name="ViTMatte",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("trimap"),
                io.Combo.Input("model", options=list(MODELS), default="vitmatte-base"),
            ],
            outputs=[io.Mask.Output("ALPHA"), io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image, trimap, model):
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        mid = MODELS[model]
        local = hf_local(mid)

        def loader(device):
            return (
                VitMatteImageProcessor.from_pretrained(local, local_files_only=True),
                VitMatteForImageMatting.from_pretrained(local, local_files_only=True).to(device).eval(),
            )

        processor, net = get_cached_model(f"vitmatte:{mid}", loader)
        device = get_torch_device()
        alphas, outs = [], []
        masks = trimap.unsqueeze(0) if trimap.ndim == 2 else trimap
        for i, pil in enumerate(tensor_to_pils(image)):
            m = masks[min(i, masks.shape[0] - 1)].detach().cpu().numpy()
            tm = np.zeros_like(m, dtype=np.uint8)
            tm[m > 0.9] = 255
            tm[(m >= 0.1) & (m <= 0.9)] = 128
            trimap_pil = Image.fromarray(tm).convert("L").resize(pil.size)
            inputs = processor(images=pil.convert("RGB"), trimaps=trimap_pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                alpha = net(**inputs).alphas[0, 0].detach().cpu().numpy()
            alphas.append(torch.from_numpy(alpha.astype(np.float32))[None, ...])
            rgb = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
            outs.append(array_to_image(rgb * alpha[..., None]))
        return io.NodeOutput(torch.cat(alphas, dim=0), torch.cat(outs, dim=0))
