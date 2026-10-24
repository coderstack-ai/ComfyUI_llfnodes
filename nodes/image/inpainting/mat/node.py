from comfy_api.latest import io

CATEGORY = "llfnodes/图像处理/图像修复"


class MATNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.MAT",
            display_name="MAT",
            category=CATEGORY,
            description="MAT 图像修复：需官方权重；未就绪时直接报错。",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.String.Input("model_path", default="", tooltip="MAT 权重路径"),
            ],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image, mask, model_path=""):
        raise RuntimeError(
            "MAT 官方推理未接入。"
            + (f" 已提供 model_path={model_path!r}，但仍缺少可导入的 MAT 实现。" if model_path else " 请安装官方 MAT 并填写 model_path。")
        )
