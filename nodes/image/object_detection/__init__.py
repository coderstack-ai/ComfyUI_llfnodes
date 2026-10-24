from .yolo import YOLONode
from .rt_detr import RTDETRNode
from .grounding_dino import GroundingDINONode

OBJECT_DETECTION_NODES = [
    YOLONode,
    RTDETRNode,
    GroundingDINONode
]

__all__ = [
    "YOLONode",
    "RTDETRNode",
    "GroundingDINONode",
    "OBJECT_DETECTION_NODES",
]
