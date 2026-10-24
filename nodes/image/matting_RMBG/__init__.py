from .birefnet import BiRefNetNode
from .bria_rmbg import BriaRMBGNode
from .modnet import MODNetNode
from .u2net import U2NetNode

MATTING_RMBG_NODES = [
    BiRefNetNode,
    BriaRMBGNode,
    MODNetNode,
    U2NetNode,
]

__all__ = [
    "BiRefNetNode",
    "BriaRMBGNode",
    "MODNetNode",
    "U2NetNode",
    "MATTING_RMBG_NODES",
]
