from .stvs import STVSNode
from .spatiotemporal_saliency import SpatiotemporalSaliencyNode

VIDEO_SALIENCY_NODES = [
    STVSNode,
    SpatiotemporalSaliencyNode
]

__all__ = [
    "STVSNode",
    "SpatiotemporalSaliencyNode",
    "VIDEO_SALIENCY_NODES",
]
