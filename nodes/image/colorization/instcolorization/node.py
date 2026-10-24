from comfy_api.latest import io

CATEGORY = "llfnodes/图像处理/黑白图像上色"


class InstColorizationNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.InstColorization",
            display_name="InstColorization",
            category=CATEGORY,
            description="InstColorization：需 Detectron2 + 官方权重；未就绪时直接报错。",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image):
        raise RuntimeError(
            "InstColorization 未内置。请安装 Detectron2 与官方权重后再用本节点；"
            "或改用已接入的 DDColor 节点。"
        )
