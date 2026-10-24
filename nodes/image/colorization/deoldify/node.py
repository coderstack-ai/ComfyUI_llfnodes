from comfy_api.latest import io

CATEGORY = "llfnodes/图像处理/黑白图像上色"


class DeOldifyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DeOldify",
            display_name="DeOldify",
            category=CATEGORY,
            description="DeOldify：需官方 DeOldify 依赖与权重；未就绪时直接报错。",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image):
        raise RuntimeError(
            "DeOldify 未内置。请安装官方 DeOldify 仓库与权重后再用本节点；"
            "或改用已接入的 DDColor 节点。"
        )
