from comfy_api.latest import io

from ..common import CATEGORY, run_rembg_session

MODELS = {
    "u2net": "u2net",
    "u2netp": "u2netp",
    "u2net_human_seg": "u2net_human_seg",
    "u2net_cloth_seg": "u2net_cloth_seg",
    "silueta": "silueta",
}
BACKGROUNDS = ["black", "white", "original"]


class U2NetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.U2Net",
            display_name="U2-Net",
            category=CATEGORY,
            description="U2-Net 前景抠图（rembg）。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="u2net"),
                io.Combo.Input("background", options=BACKGROUNDS, default="black"),
            ],
            outputs=[
                io.Image.Output("IMAGE"),
                io.Mask.Output("MASK"),
            ],
        )

    @classmethod
    def execute(cls, image, model, background):
        img, mask = run_rembg_session(MODELS[model], image, background=background)
        return io.NodeOutput(img, mask)
