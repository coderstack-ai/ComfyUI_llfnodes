from .sam2 import SAM2Node
from .xmem import XMemNode
from .cutie import CutieNode
from .deva import DEVANode

VIDEO_OBJECT_SEGMENTATION_NODES = [
    SAM2Node,
    XMemNode,
    CutieNode,
    DEVANode
]

__all__ = [
    "SAM2Node",
    "XMemNode",
    "CutieNode",
    "DEVANode",
    "VIDEO_OBJECT_SEGMENTATION_NODES",
]
