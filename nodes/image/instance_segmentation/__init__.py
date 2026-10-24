from .sam import SAMNode
from .sam2 import SAM2Node
from .mask2former import Mask2FormerInstanceNode
from .mask_rcnn import MaskRCNNNode

INSTANCE_SEGMENTATION_NODES = [
    SAMNode,
    SAM2Node,
    Mask2FormerInstanceNode,
    MaskRCNNNode,
]

__all__ = [
    "SAMNode",
    "SAM2Node",
    "Mask2FormerInstanceNode",
    "MaskRCNNNode",
    "INSTANCE_SEGMENTATION_NODES",
]
