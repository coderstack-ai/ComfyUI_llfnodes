from .depth_anything_v2 import DepthAnythingV2Node
from .depth_anything_v3 import DepthAnythingV3Node
from .midas import MiDaSNode
from .zoedepth import ZoeDepthNode
from .marigold import MarigoldNode
from .metric3d import Metric3DNode

DEPTH_ESTIMATION_NODES = [
    DepthAnythingV2Node,
    DepthAnythingV3Node,
    MiDaSNode,
    ZoeDepthNode,
    MarigoldNode,
    Metric3DNode,
]

__all__ = [
    "DepthAnythingV2Node",
    "DepthAnythingV3Node",
    "MiDaSNode",
    "ZoeDepthNode",
    "MarigoldNode",
    "Metric3DNode",
    "DEPTH_ESTIMATION_NODES",
]
