from comfy_api.latest import io
from ...cv_shared import array_to_image, get_cached_model, require_pkg, tensor_to_pils
import numpy as np
import torch

CATEGORY = "llfnodes/图像处理/精细透明度估计"


class FineMODNetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.FineMODNet",
            display_name="MODNet",
            category=CATEGORY,
            inputs=[io.Image.Input("image")],
            outputs=[io.Mask.Output("ALPHA"), io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image):
        require_pkg("rembg", "pip install rembg")
        from rembg import new_session, remove

        session = get_cached_model("rembg:modnet", lambda d: new_session("modnet"))
        alphas, outs = [], []
        for pil in tensor_to_pils(image):
            rgba = remove(pil.convert("RGB"), session=session)
            arr = np.asarray(rgba)
            alpha = arr[..., 3].astype(np.float32) / 255.0
            alphas.append(torch.from_numpy(alpha)[None, ...])
            outs.append(array_to_image(arr[..., :3]))
        return io.NodeOutput(torch.cat(alphas, dim=0), torch.cat(outs, dim=0))
