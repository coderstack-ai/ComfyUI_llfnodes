from comfy_api.latest import io

CATEGORY = "llfnodes/图像处理/线段与几何结构检测"


class HAWPNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.HAWP",
            display_name="HAWP",
            category=CATEGORY,
            description="HAWP 线框检测。需官方 hawp 依赖与权重；未就绪时直接报错。",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("model_path", default="", tooltip="HAWP 权重路径"),
                io.Boolean.Input("draw_on_black", default=True, label_on="黑底", label_off="原图"),
            ],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image, model_path="", draw_on_black=True):
        raise RuntimeError(
            "HAWP 官方推理未接入。"
            "请按 https://github.com/cherubicXN/hawp 安装并填写 model_path。"
            + (f" 当前 model_path={model_path!r}。" if model_path else "")
        )
