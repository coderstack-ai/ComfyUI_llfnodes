from .dsine import DSINENode
from .normalbae import NormalBaeNode
from .marigold_normals import MarigoldNormalsNode

NORMAL_ESTIMATION_NODES = [
    DSINENode,
    NormalBaeNode,
    MarigoldNormalsNode,
]

__all__ = [
    "DSINENode",
    "NormalBaeNode",
    "MarigoldNormalsNode",
    "NORMAL_ESTIMATION_NODES",
]
