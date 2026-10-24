from .basicvsrpp import BasicVSRPlusPlusNode
from .realbasicvsr import RealBasicVSRNode
from .rvrt import VSRRVRTNode

VIDEO_SUPER_RESOLUTION_NODES = [
    BasicVSRPlusPlusNode,
    RealBasicVSRNode,
    VSRRVRTNode
]

__all__ = [
    "BasicVSRPlusPlusNode",
    "RealBasicVSRNode",
    "VSRRVRTNode",
    "VIDEO_SUPER_RESOLUTION_NODES",
]
