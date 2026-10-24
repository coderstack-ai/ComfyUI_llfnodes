from comfy_api.latest import io
from ...cv_shared import get_torch_device, require_pkg, tensor_to_pils, pils_to_tensor
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

CATEGORY = "llfnodes/图像处理/图像超分辨率"


class HATNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.HAT",
            display_name="HAT",
            category=CATEGORY,
            description="通过 spandrel 加载权重（models/upscale_models）。",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("model_name", default="HAT_SRx4.pth"),
            ],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image, model_name):
        require_pkg("spandrel", "pip install spandrel")
        import folder_paths
        from pathlib import Path
        from spandrel import ModelLoader

        path = Path(folder_paths.get_full_path_or_raise("upscale_models", model_name))
        device = get_torch_device()
        model = ModelLoader().load_from_file(str(path)).model.eval().to(device)
        outs = []
        for pil in tensor_to_pils(image):
            t = to_tensor(pil).unsqueeze(0).to(device)
            with torch.inference_mode():
                out = model(t)
            outs.append(to_pil_image(out.squeeze(0).clamp(0, 1).cpu()))
        return io.NodeOutput(pils_to_tensor(outs))
