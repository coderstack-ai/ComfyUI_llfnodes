from comfy_api.latest import io

CATEGORY = "llfnodes/图像处理/单目三维重建"


class DUSt3RNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DUSt3R",
            display_name="DUSt3R",
            category=CATEGORY,
            description="DUSt3R：需官方仓库与权重；未就绪时直接报错。",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("model_path", default="", tooltip="DUSt3R 权重/仓库路径"),
            ],
            outputs=[io.Image.Output("DEPTH")],
        )

    @classmethod
    def execute(cls, image, model_path=""):
        raise RuntimeError(
            "DUSt3R 未安装。请按 https://github.com/naver/dust3r 安装并填写 model_path。"
            + (f" 当前 model_path={model_path!r}。" if model_path else "")
        )
