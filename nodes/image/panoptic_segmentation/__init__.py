from .mask2former import PanopticMask2FormerNode
from .oneformer import PanopticOneFormerNode

PANOPTIC_SEGMENTATION_NODES = [
    PanopticMask2FormerNode,
    PanopticOneFormerNode
]

__all__ = [
    "PanopticMask2FormerNode",
    "PanopticOneFormerNode",
    "PANOPTIC_SEGMENTATION_NODES",
]
