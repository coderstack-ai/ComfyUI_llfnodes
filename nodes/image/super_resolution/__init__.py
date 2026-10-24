from .realesrgan import RealESRGANNode
from .swinir import SwinIRNode
from .hat import HATNode

SUPER_RESOLUTION_NODES = [
    RealESRGANNode,
    SwinIRNode,
    HATNode
]

__all__ = [
    "RealESRGANNode",
    "SwinIRNode",
    "HATNode",
    "SUPER_RESOLUTION_NODES",
]
