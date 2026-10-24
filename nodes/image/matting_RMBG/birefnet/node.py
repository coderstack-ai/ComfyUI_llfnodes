from comfy_api.latest import io

from ..common import CATEGORY, run_birefnet_hf, run_rembg_session

MODELS = {
    "BiRefNet": "ZhengPeng7/BiRefNet",
    "BiRefNet-HR": "ZhengPeng7/BiRefNet_HR",
    "BiRefNet-portrait": "ZhengPeng7/BiRefNet-portrait",
    "rembg-general": "birefnet-general",
    "rembg-general-lite": "birefnet-general-lite",
    "rembg-portrait": "birefnet-portrait",
}
BACKGROUNDS = ["black", "white", "original"]


class BiRefNetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.BiRefNet",
            display_name="BiRefNet",
            category=CATEGORY,
            description="BiRefNet 前景抠图。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="BiRefNet"),
                io.Combo.Input("background", options=BACKGROUNDS, default="black"),
            ],
            outputs=[
                io.Image.Output("IMAGE"),
                io.Mask.Output("MASK"),
            ],
        )

    @classmethod
    def execute(cls, image, model, background):
        model_id = MODELS[model]
        if model_id.startswith("birefnet"):
            img, mask = run_rembg_session(model_id, image, background=background)
        else:
            img, mask = run_birefnet_hf(model_id, image, background=background)
        return io.NodeOutput(img, mask)
