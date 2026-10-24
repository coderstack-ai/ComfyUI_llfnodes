from .deoldify import DeOldifyNode
from .ddcolor import DDColorNode
from .instcolorization import InstColorizationNode

COLORIZATION_NODES = [
    DeOldifyNode,
    DDColorNode,
    InstColorizationNode
]

__all__ = [
    "DeOldifyNode",
    "DDColorNode",
    "InstColorizationNode",
    "COLORIZATION_NODES",
]
