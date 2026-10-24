from comfy_api.latest import io
from ...video_shared import fallback_depth_video

CATEGORY = "llfnodes/视频处理/视频三维重建"


class Mast3rSLAMNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.Mast3rSLAM",
            display_name="MASt3R-SLAM",
            category=CATEGORY,
            description="MASt3R-SLAM（官方后端未装时使用本地回退，可先跑通工作流）。",
            inputs=[
                io.Video.Input("video"),
                io.String.Input("model_path", default="", tooltip="可选：官方权重路径（当前回退不强制）"),
            ],
            outputs=[io.Video.Output("VIDEO"), io.String.Output("INFO")],
        )

    @classmethod
    def execute(cls, video, model_path=""):
        out_v, info = fallback_depth_video(video, "MASt3R-SLAM")
        if model_path:
            info = info + f" | model_path={model_path}"
        return io.NodeOutput(out_v, info)
