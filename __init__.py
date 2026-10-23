from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from .nodes import (
    ImageLoadNode,
    ChannelSplitNode,
    ChannelMergeNode,
    ImagePreviewNode,
    AudioLoadNode,
    VideoLoadNode,
)


class LlfNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ImageLoadNode,
            ChannelSplitNode,
            ChannelMergeNode,
            ImagePreviewNode,
            AudioLoadNode,
            VideoLoadNode,
        ]


async def comfy_entrypoint() -> LlfNodesExtension:
    return LlfNodesExtension()
