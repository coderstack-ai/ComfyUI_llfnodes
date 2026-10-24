from comfy_api.latest import io

from ..ddcolor.node import DDColorNode

CATEGORY = "llfnodes/图像处理/黑白图像上色"


class InstColorizationNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.InstColorization",
            display_name="InstColorization",
            category=CATEGORY,
            description="InstColorization 需 Detectron2；当前回退为 DDColor LAB 近似上色。",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image):
        return DDColorNode.execute(image)
