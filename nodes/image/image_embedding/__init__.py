from .clip import CLIPEmbedNode
from .siglip import SigLIPEmbedNode
from .dinov2 import DINOv2EmbedNode

IMAGE_EMBEDDING_NODES = [
    CLIPEmbedNode,
    SigLIPEmbedNode,
    DINOv2EmbedNode
]

__all__ = [
    "CLIPEmbedNode",
    "SigLIPEmbedNode",
    "DINOv2EmbedNode",
    "IMAGE_EMBEDDING_NODES",
]
