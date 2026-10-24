from .segformer import SegFormerNode
from .oneformer import OneFormerNode
from .mask2former import Mask2FormerNode
from .upernet import UPerNetNode
from .uniformer import UniFormerNode

SEMANTIC_SEGMENTATION_NODES = [
    SegFormerNode,
    OneFormerNode,
    Mask2FormerNode,
    UPerNetNode,
    UniFormerNode,
]

__all__ = [
    "SegFormerNode",
    "OneFormerNode",
    "Mask2FormerNode",
    "UPerNetNode",
    "UniFormerNode",
    "SEMANTIC_SEGMENTATION_NODES",
]
