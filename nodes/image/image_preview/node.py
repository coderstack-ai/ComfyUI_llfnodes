import torch
from comfy_api.latest import io, ui


def _to_preview_image(data: torch.Tensor) -> torch.Tensor:
    """Normalize MASK / 1ch / 3ch / 4ch tensors to a previewable IMAGE batch."""
    if data.ndim == 2:
        data = data.unsqueeze(0)

    # MASK: [B, H, W]
    if data.ndim == 3:
        return data.reshape((-1, 1, data.shape[-2], data.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

    if data.ndim != 4:
        raise ValueError(f"Unsupported tensor shape for preview: {tuple(data.shape)}")

    channels = data.shape[-1]
    if channels == 1:
        return data.expand(-1, -1, -1, 3)
    if channels in (3, 4):
        return data
    raise ValueError(f"Unsupported channel count: {channels}, expected 1 / 3 / 4")


class ImagePreviewNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.ImagePreviewNode",
            display_name="图像显示",
            category="llfnodes/image",
            description="预览图像，支持单通道（MASK/灰度）、3 通道（RGB）、4 通道（RGBA）。",
            is_output_node=True,
            inputs=[
                io.MultiType.Input("image", types=[io.Image, io.Mask]),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, image):
        preview = _to_preview_image(image)
        return io.NodeOutput(ui=ui.PreviewImage(preview, cls=cls))
