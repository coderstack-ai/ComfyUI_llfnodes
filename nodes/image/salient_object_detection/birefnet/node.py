from comfy_api.latest import io
from ...cv_shared import array_to_image, get_cached_model, tensor_to_pils
import numpy as np
import torch

CATEGORY = "llfnodes/图像处理/显著性检测"
MODELS = {'birefnet-general': 'birefnet-general', 'birefnet-general-lite': 'birefnet-general-lite', 'birefnet-portrait': 'birefnet-portrait'}


class SalientBiRefNetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.SalientBiRefNet",
            display_name="BiRefNet",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS), default="birefnet-general"),
            ],
            outputs=[io.Image.Output("IMAGE"), io.Mask.Output("MASK")],
        )

    @classmethod
    def execute(cls, image, model):
        from rembg import new_session, remove

        session = get_cached_model(f"salient-rembg:{MODELS[model]}", lambda d: new_session(MODELS[model]))
        imgs, masks = [], []
        for pil in tensor_to_pils(image):
            rgba = remove(pil, session=session).convert("RGBA")
            alpha = np.asarray(rgba.split()[-1], dtype=np.float32) / 255.0
            rgb = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
            out = rgb * alpha[..., None]
            imgs.append(torch.from_numpy(out)[None, ...])
            masks.append(torch.from_numpy(alpha)[None, ...])
        return io.NodeOutput(torch.cat(imgs, dim=0), torch.cat(masks, dim=0))
