from comfy_api.latest import io

from ..schp.node import SCHPNode

CATEGORY = "llfnodes/图像处理/人体密集姿态与人体解析"


class DensePoseNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DensePose",
            display_name="DensePose",
            category=CATEGORY,
            description="DensePose 需 detectron2；当前回退为 SCHP/SegFormer 人体解析可视化。",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image):
        out = SCHPNode.execute(image, "segformer-b2-clothes")
        return io.NodeOutput(out.args[0])
