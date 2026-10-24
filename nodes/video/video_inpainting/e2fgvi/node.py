from comfy_api.latest import io
from ...video_shared import fallback_passthrough

CATEGORY = "llfnodes/视频处理/视频修复"


class E2FGVINode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.E2FGVI",
            display_name="E2FGVI",
            category=CATEGORY,
            description="E2FGVI（官方后端未装时使用本地回退，可先跑通工作流）。",
            inputs=[
                io.Video.Input("video"),
                io.String.Input("model_path", default="", tooltip="可选：官方权重路径（当前回退不强制）"),
            ],
            outputs=[io.Video.Output("VIDEO"), io.String.Output("INFO")],
        )

    @classmethod
    def execute(cls, video, model_path=""):
        out_v, info = fallback_passthrough(video, "E2FGVI")
        if model_path:
            info = info + f" | model_path={model_path}"
        return io.NodeOutput(out_v, info)
