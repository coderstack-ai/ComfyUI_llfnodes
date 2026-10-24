from comfy_api.latest import io
from ...cv_shared import array_to_image, get_cached_model, get_torch_device, tensor_to_pils
import numpy as np
import torch
from ....model_store import hf_local

CATEGORY = "llfnodes/图像处理/材质与固有图像分解"
MODELS = {"v1.1": "prs-eth/marigold-iid-appearance-v1-1", "lighting-v1.1": "prs-eth/marigold-iid-lighting-v1-1"}


class MarigoldIntrinsicNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.MarigoldIntrinsic",
            display_name="Marigold Intrinsic",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS), default="v1.1"),
                io.Int.Input("steps", default=4, min=1, max=50),
            ],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image, model, steps):
        try:
            from diffusers import MarigoldIntrinsicsPipeline
        except Exception as e:
            raise RuntimeError(
                "无法导入 MarigoldIntrinsicsPipeline，请升级 diffusers/huggingface_hub。\n" + str(e)
            ) from e

        mid = MODELS[model]
        local = hf_local(mid, folder_name="diffusers")

        def loader(device):
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            return MarigoldIntrinsicsPipeline.from_pretrained(local, torch_dtype=dtype, local_files_only=True).to(device)

        pipe = get_cached_model(f"marigold-iid:{mid}", loader)
        outs = []
        for pil in tensor_to_pils(image):
            out = pipe(pil, num_inference_steps=steps, match_input_resolution=True, output_type="np")
            pred = np.squeeze(np.asarray(out.prediction, dtype=np.float32))
            if pred.ndim == 3 and pred.shape[0] in (1, 3):
                pred = np.transpose(pred, (1, 2, 0))
            if pred.ndim == 2:
                pred = np.repeat(pred[..., None], 3, axis=-1)
            if pred.shape[-1] > 3:
                pred = pred[..., :3]
            outs.append(array_to_image(pred))
        return io.NodeOutput(torch.cat(outs, dim=0))
