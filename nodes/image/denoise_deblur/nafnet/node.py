from comfy_api.latest import io
from ...cv_shared import get_torch_device, require_pkg, tensor_to_pils, pils_to_tensor
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

CATEGORY = "llfnodes/图像处理/图像去噪与去模糊"


class NAFNetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.NAFNet",
            display_name="NAFNet",
            category=CATEGORY,
            description="通过 spandrel 加载 {model_name}。",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("model_name", default="NAFNet.pth"),
            ],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image, model_name):
        require_pkg("spandrel", "pip install spandrel")
        import folder_paths
        from pathlib import Path
        from spandrel import ModelLoader

        try:
            path = Path(folder_paths.get_full_path_or_raise("upscale_models", model_name))
        except Exception:
            path = Path(model_name)
            if not path.exists():
                raise RuntimeError(f"找不到模型: {model_name}，请放到 ComfyUI/models/upscale_models/")
        device = get_torch_device()
        model = ModelLoader().load_from_file(str(path)).model.eval().to(device)
        outs = []
        for pil in tensor_to_pils(image):
            t = to_tensor(pil).unsqueeze(0).to(device)
            with torch.inference_mode():
                out = model(t)
            outs.append(to_pil_image(out.squeeze(0).clamp(0, 1).cpu()))
        return io.NodeOutput(pils_to_tensor(outs))
