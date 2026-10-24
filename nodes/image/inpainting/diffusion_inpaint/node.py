from comfy_api.latest import io
from ...cv_shared import get_cached_model, get_torch_device, tensor_to_pils, pils_to_tensor
import numpy as np
from PIL import Image
from ....model_store import hf_local

CATEGORY = "llfnodes/图像处理/图像修复"
MODELS = {
    "sd15-inpaint": "runwayml/stable-diffusion-inpainting",
    "sdxl-inpaint": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
}


class DiffusionInpaintNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DiffusionInpaint",
            display_name="扩散修复模型",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.String.Input("prompt", default="high quality", multiline=True),
                io.Combo.Input("model", options=list(MODELS), default="sd15-inpaint"),
                io.Int.Input("steps", default=20, min=1, max=100),
            ],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image, mask, prompt, model, steps):
        from diffusers import StableDiffusionInpaintPipeline
        import torch

        mid = MODELS[model]
        local = hf_local(mid, folder_name="diffusers")

        def loader(device):
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            return StableDiffusionInpaintPipeline.from_pretrained(local, torch_dtype=dtype, local_files_only=True).to(device)

        pipe = get_cached_model(f"inpaint:{mid}", loader)
        outs = []
        masks = mask.unsqueeze(0) if mask.ndim == 2 else mask
        for i, pil in enumerate(tensor_to_pils(image)):
            m = masks[min(i, masks.shape[0] - 1)].detach().cpu().numpy()
            mask_pil = Image.fromarray((m * 255).astype(np.uint8)).convert("L").resize(pil.size)
            result = pipe(
                prompt=prompt,
                image=pil.convert("RGB"),
                mask_image=mask_pil,
                num_inference_steps=steps,
            ).images[0]
            outs.append(result)
        return io.NodeOutput(pils_to_tensor(outs))
