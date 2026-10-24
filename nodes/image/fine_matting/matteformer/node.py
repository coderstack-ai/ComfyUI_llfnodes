from comfy_api.latest import io

CATEGORY = "llfnodes/图像处理/精细透明度估计"


class MatteFormerNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.MatteFormer",
            display_name="MatteFormer",
            category=CATEGORY,
            description="MatteFormer：需官方权重；未就绪时直接报错。",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("trimap"),
                io.String.Input("model_path", default="", tooltip="MatteFormer 权重路径"),
            ],
            outputs=[io.Mask.Output("ALPHA")],
        )

    @classmethod
    def execute(cls, image, trimap, model_path=""):
        raise RuntimeError(
            "MatteFormer 官方推理未接入。"
            + (f" 已提供 model_path={model_path!r}，但仍缺少可导入的实现。" if model_path else " 请安装官方 MatteFormer 并填写 model_path。")
        )
