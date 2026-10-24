from comfy_api.latest import io

from ..common import CATEGORY, run_birefnet_hf, run_rembg_session

# RMBG-2.0 为 gated（需 HF 同意协议 + login）；默认用可公开下载的 1.4。
MODELS = {
    "RMBG-1.4": "briaai/RMBG-1.4",
    "RMBG-2.0": "briaai/RMBG-2.0",
    "rembg-bria-rmbg": "bria-rmbg",
}
BACKGROUNDS = ["black", "white", "original"]


class BriaRMBGNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.BriaRMBG",
            display_name="BRIA RMBG",
            category=CATEGORY,
            description="BRIA RMBG 抠图。RMBG-2.0 需 HF 授权；默认 RMBG-1.4 可直接下载。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="RMBG-1.4"),
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
        if model_id == "bria-rmbg":
            img, mask = run_rembg_session(model_id, image, background=background)
        else:
            img, mask = run_birefnet_hf(model_id, image, background=background)
        return io.NodeOutput(img, mask)
