from comfy_api.latest import io

CATEGORY = "llfnodes/图像处理/黑白图像上色"


class DDColorNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DDColor",
            display_name="DDColor",
            category=CATEGORY,
            description="DDColor：需官方权重/实现；未就绪时直接报错。",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("model_path", default="", tooltip="DDColor 权重路径"),
            ],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image, model_path=""):
        raise RuntimeError(
            "DDColor 官方推理未接入本节点。"
            + (f" 已提供 model_path={model_path!r}，但仍缺少可导入的 DDColor 实现。" if model_path else " 请安装官方 DDColor 并填写 model_path。")
        )
