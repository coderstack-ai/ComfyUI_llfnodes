from comfy_api.latest import io
from ...video_shared import missing_backend

CATEGORY = "llfnodes/视频处理/视频修复"


class ProPainterNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.ProPainter",
            display_name="ProPainter",
            category=CATEGORY,
            description="ProPainter：需官方依赖/权重；未就绪时直接报错，不做替代实现。",
            inputs=[
                io.Video.Input("video"),
                io.String.Input("model_path", default="", tooltip="官方权重或仓库路径（必填，视具体后端而定）"),
            ],
            outputs=[io.Video.Output("VIDEO"), io.String.Output("INFO")],
        )

    @classmethod
    def execute(cls, video, model_path=""):
        tip = "安装 ProPainter/E2FGVI/STTN 官方实现并填写 model_path。"
        if model_path:
            tip = tip + " 当前 model_path=" + repr(model_path)
        else:
            tip = tip + " 当前未提供 model_path。"
        missing_backend("ProPainter", tip)
