from comfy_api.latest import io

CATEGORY = "llfnodes/图像处理/人体密集姿态与人体解析"


class DensePoseNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DensePose",
            display_name="DensePose",
            category=CATEGORY,
            description="DensePose：需 detectron2 + densepose；未安装时直接报错。",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image):
        raise RuntimeError(
            "DensePose 需要 detectron2 与 densepose 扩展，当前环境未就绪。"
            "请安装官方依赖后再用本节点；人体解析可改用 SCHP / SegFormer Human 节点。"
        )
