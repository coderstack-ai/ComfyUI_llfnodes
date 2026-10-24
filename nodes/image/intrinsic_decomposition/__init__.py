from .marigold_intrinsic import MarigoldIntrinsicNode
from .intrinsic_image import IntrinsicImageNode

INTRINSIC_DECOMPOSITION_NODES = [
    MarigoldIntrinsicNode,
    IntrinsicImageNode
]

__all__ = [
    "MarigoldIntrinsicNode",
    "IntrinsicImageNode",
    "INTRINSIC_DECOMPOSITION_NODES",
]
