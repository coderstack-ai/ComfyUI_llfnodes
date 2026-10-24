from comfy_api.latest import io
from ...video_shared import fallback_denoise_temporal

CATEGORY = "llfnodes/视频处理/视频去噪"


class FastDVDnetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.FastDVDnet",
            display_name="FastDVDnet",
            category=CATEGORY,
            description="FastDVDnet（官方后端未装时使用本地回退，可先跑通工作流）。",
            inputs=[
                io.Video.Input("video"),
                io.String.Input("model_path", default="", tooltip="可选：官方权重路径（当前回退不强制）"),
            ],
            outputs=[io.Video.Output("VIDEO"), io.String.Output("INFO")],
        )

    @classmethod
    def execute(cls, video, model_path=""):
        out_v, info = fallback_denoise_temporal(video, "FastDVDnet")
        if model_path:
            info = info + f" | model_path={model_path}"
        return io.NodeOutput(out_v, info)
