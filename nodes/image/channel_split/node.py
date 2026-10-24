import torch
from comfy_api.latest import io


class ChannelSplitNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.ChannelSplitNode",
            display_name="通道拆分",
            category="llfnodes/图像处理",
            description="将图像拆分为 R / G / B / A 四个通道（MASK）。",
            inputs=[
                io.Image.Input("image"),
            ],
            outputs=[
                io.Mask.Output("R"),
                io.Mask.Output("G"),
                io.Mask.Output("B"),
                io.Mask.Output("A"),
            ],
        )

    @classmethod
    def execute(cls, image):
        r = image[:, :, :, 0]
        g = image[:, :, :, 1]
        b = image[:, :, :, 2]
        if image.shape[-1] > 3:
            a = image[:, :, :, 3]
        else:
            a = torch.ones(image.shape[:-1], dtype=image.dtype, device=image.device)
        return io.NodeOutput(r, g, b, a)
