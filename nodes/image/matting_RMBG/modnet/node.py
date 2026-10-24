from comfy_api.latest import io

from ..common import CATEGORY, run_modnet

BACKGROUNDS = ["black", "white", "original"]


class MODNetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.MODNet",
            display_name="MODNet",
            category=CATEGORY,
            description="MODNet 人像抠图（ONNX）。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("background", options=BACKGROUNDS, default="black"),
            ],
            outputs=[
                io.Image.Output("IMAGE"),
                io.Mask.Output("MASK"),
            ],
        )

    @classmethod
    def execute(cls, image, background):
        img, mask = run_modnet(image, background=background)
        return io.NodeOutput(img, mask)
