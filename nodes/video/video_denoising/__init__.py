from .fastdvdnet import FastDVDnetNode
from .vrt import DenoiseVRTNode
from .rvrt import DenoiseRVRTNode

VIDEO_DENOISING_NODES = [
    FastDVDnetNode,
    DenoiseVRTNode,
    DenoiseRVRTNode
]

__all__ = [
    "FastDVDnetNode",
    "DenoiseVRTNode",
    "DenoiseRVRTNode",
    "VIDEO_DENOISING_NODES",
]
