from comfy_api.latest import io

CATEGORY = "llfnodes/图像处理/图像文字识别"


class DBNetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DBNet",
            display_name="DBNet",
            category=CATEGORY,
            description="DBNet：需官方权重/实现；未就绪时直接报错。",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("model_path", default="", tooltip="DBNet 权重路径"),
            ],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image, model_path=""):
        raise RuntimeError(
            "DBNet 官方推理未接入。"
            + (f" 已提供 model_path={model_path!r}，但仍缺少可导入的实现。" if model_path else " 请安装官方 DBNet 并填写 model_path。")
        )
