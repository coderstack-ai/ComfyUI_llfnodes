import torch
from comfy_api.latest import io, ui


def _to_preview_image(data: torch.Tensor) -> torch.Tensor:
    """Normalize MASK / 1ch / 3ch / 4ch tensors to a previewable IMAGE batch."""
    if not isinstance(data, torch.Tensor):
        raise ValueError(f"Unsupported preview input type: {type(data)}")

    if data.ndim == 2:
        data = data.unsqueeze(0)

    # MASK: [B, H, W]
    if data.ndim == 3:
        return data.reshape((-1, 1, data.shape[-2], data.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

    # Nested batch e.g. [B, N, H, W, C] / [T, H, W, C] already covered by ndim==4
    # Collapse leading dims: (1, 2, 1024, 1024, 3) -> (2, 1024, 1024, 3)
    if data.ndim > 4:
        data = data.reshape((-1, data.shape[-3], data.shape[-2], data.shape[-1]))

    if data.ndim != 4:
        raise ValueError(f"Unsupported tensor shape for preview: {tuple(data.shape)}")

    channels = data.shape[-1]
    if channels == 1:
        return data.expand(-1, -1, -1, 3)
    if channels in (3, 4):
        return data
    # NCHW-ish: [B, C, H, W]
    if data.shape[1] in (1, 3, 4) and data.shape[-1] not in (1, 3, 4):
        data = data.movedim(1, -1)
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
            category="llfnodes/图像处理",
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
