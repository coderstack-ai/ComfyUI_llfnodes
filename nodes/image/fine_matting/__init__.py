from .vitmatte import ViTMatteNode
from .modnet import FineMODNetNode
from .matteformer import MatteFormerNode

FINE_MATTING_NODES = [
    ViTMatteNode,
    FineMODNetNode,
    MatteFormerNode
]

__all__ = [
    "ViTMatteNode",
    "FineMODNetNode",
    "MatteFormerNode",
    "FINE_MATTING_NODES",
]
