from .rife import RIFENode
from .film import FILMNode
from .amt import AMTNode
from .ifrnet import IFRNetNode

FRAME_INTERPOLATION_NODES = [
    RIFENode,
    FILMNode,
    AMTNode,
    IFRNetNode
]

__all__ = [
    "RIFENode",
    "FILMNode",
    "AMTNode",
    "IFRNetNode",
    "FRAME_INTERPOLATION_NODES",
]
