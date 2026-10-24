from comfy_api.latest import io
from ...cv_shared import require_pkg, tensor_to_pils, pils_to_tensor
import numpy as np
from PIL import Image

CATEGORY = "llfnodes/图像处理/图像修复"


class LaMaNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.LaMa",
            display_name="LaMa",
            category=CATEGORY,
            description="simple-lama-inpainting 封装。",
            inputs=[io.Image.Input("image"), io.Mask.Input("mask")],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image, mask):
        require_pkg("simple_lama_inpainting", "pip install simple-lama-inpainting")
        from simple_lama_inpainting import SimpleLama

        lama = SimpleLama()
        outs = []
        masks = mask.unsqueeze(0) if mask.ndim == 2 else mask
        for i, pil in enumerate(tensor_to_pils(image)):
            m = masks[min(i, masks.shape[0] - 1)].detach().cpu().numpy()
            mask_pil = Image.fromarray((m * 255).astype(np.uint8)).convert("L").resize(pil.size)
            outs.append(lama(pil.convert("RGB"), mask_pil))
        return io.NodeOutput(pils_to_tensor(outs))
