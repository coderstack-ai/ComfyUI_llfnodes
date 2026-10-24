from comfy_api.latest import io

from ...depth_estimation.common import run_hf_depth

CATEGORY = "llfnodes/图像处理/单目三维重建"

# 官方 vggt 不在 PyPI。用 Depth Anything 作可运行深度预览。
_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


class VGGTNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.VGGT",
            display_name="VGGT",
            category=CATEGORY,
            description=(
                "单目深度预览。完整 VGGT 需 GitHub 安装；"
                "当前用 Depth Anything V2 输出深度图。"
            ),
            inputs=[
                io.Image.Input("image"),
                io.Boolean.Input("invert", default=False, label_on="near=black", label_off="near=white"),
            ],
            outputs=[io.Image.Output("DEPTH")],
        )

    @classmethod
    def execute(cls, image, invert=False):
        return io.NodeOutput(run_hf_depth(_DEPTH_MODEL, image, invert=invert))
