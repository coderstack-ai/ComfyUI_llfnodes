from .lama import LaMaNode
from .mat import MATNode
from .diffusion_inpaint import DiffusionInpaintNode

INPAINTING_NODES = [
    LaMaNode,
    MATNode,
    DiffusionInpaintNode
]

__all__ = [
    "LaMaNode",
    "MATNode",
    "DiffusionInpaintNode",
    "INPAINTING_NODES",
]
