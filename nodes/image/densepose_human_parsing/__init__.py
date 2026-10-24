from .densepose import DensePoseNode
from .schp import SCHPNode
from .segformer_human import SegFormerHumanNode

DENSEPOSE_HUMAN_PARSING_NODES = [
    DensePoseNode,
    SCHPNode,
    SegFormerHumanNode
]

__all__ = [
    "DensePoseNode",
    "SCHPNode",
    "SegFormerHumanNode",
    "DENSEPOSE_HUMAN_PARSING_NODES",
]
