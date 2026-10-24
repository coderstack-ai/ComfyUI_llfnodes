from comfy_api.latest import io

CATEGORY = "llfnodes/图像处理/透视与消失点估计"


class GeoCalibNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.GeoCalib",
            display_name="GeoCalib",
            category=CATEGORY,
            description="GeoCalib：需官方 geocalib 包；未安装时直接报错。",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output("IMAGE"), io.String.Output("PARAMS")],
        )

    @classmethod
    def execute(cls, image):
        try:
            import geocalib  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "GeoCalib 未安装。请按官方仓库安装 geocalib 后再用本节点。"
            ) from e
        raise RuntimeError("已检测到 geocalib 包，但本节点尚未接入官方推理调用。请等待完整实现或直接用官方 API。")
