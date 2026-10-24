from .u2net import SalientU2NetNode
from .isnet import SalientISNetNode
from .birefnet import SalientBiRefNetNode

SALIENT_OBJECT_DETECTION_NODES = [
    SalientU2NetNode,
    SalientISNetNode,
    SalientBiRefNetNode
]

__all__ = [
    "SalientU2NetNode",
    "SalientISNetNode",
    "SalientBiRefNetNode",
    "SALIENT_OBJECT_DETECTION_NODES",
]
