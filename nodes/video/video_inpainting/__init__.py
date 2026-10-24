from .propainter import ProPainterNode
from .e2fgvi import E2FGVINode
from .sttn import STTNNode

VIDEO_INPAINTING_NODES = [
    ProPainterNode,
    E2FGVINode,
    STTNNode
]

__all__ = [
    "ProPainterNode",
    "E2FGVINode",
    "STTNNode",
    "VIDEO_INPAINTING_NODES",
]
