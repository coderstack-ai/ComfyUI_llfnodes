from .hed import HEDNode
from .pidinet import PiDiNetNode
from .teed import TEEDNode
from .anyline import AnyLineNode

EDGE_LINEART_NODES = [
    HEDNode,
    PiDiNetNode,
    TEEDNode,
    AnyLineNode,
]

__all__ = [
    "HEDNode",
    "PiDiNetNode",
    "TEEDNode",
    "AnyLineNode",
    "EDGE_LINEART_NODES",
]
