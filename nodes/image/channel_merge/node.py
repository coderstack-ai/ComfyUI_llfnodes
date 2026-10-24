import torch
from comfy_api.latest import io


def _as_mask_batch(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    if mask.ndim == 3:
        return mask
    if mask.ndim == 4 and mask.shape[-1] == 1:
        return mask[..., 0]
    raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")


class ChannelMergeNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.ChannelMergeNode",
            display_name="通道合并",
            category="llfnodes/图像处理",
            description="将 R / G / B（可选 A）通道合并为图像。不接 A 时输出 3 通道，接 A 时输出 4 通道。",
            inputs=[
                io.Mask.Input("R"),
                io.Mask.Input("G"),
                io.Mask.Input("B"),
                io.Mask.Input("A", optional=True),
            ],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, R, G, B, A=None):
        r = _as_mask_batch(R)
        g = _as_mask_batch(G)
        b = _as_mask_batch(B)
        channels = [r, g, b]
        if A is not None:
            channels.append(_as_mask_batch(A))

        batch = max(c.shape[0] for c in channels)
        height = max(c.shape[1] for c in channels)
        width = max(c.shape[2] for c in channels)

        aligned = []
        for c in channels:
            if c.shape[0] == 1 and batch > 1:
                c = c.expand(batch, -1, -1)
            if c.shape[1:] != (height, width):
                c = torch.nn.functional.interpolate(
                    c.unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            if c.shape[0] != batch:
                raise ValueError(f"Channel batch size mismatch: got {c.shape[0]}, expected {batch}")
            aligned.append(c)

        image = torch.stack(aligned, dim=-1).clamp(0.0, 1.0)
        return io.NodeOutput(image)
