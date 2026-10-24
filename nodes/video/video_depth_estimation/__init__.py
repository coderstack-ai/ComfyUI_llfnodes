from .video_depth_anything import VideoDepthAnythingNode
from .depthcrafter import DepthCrafterNode

VIDEO_DEPTH_ESTIMATION_NODES = [
    VideoDepthAnythingNode,
    DepthCrafterNode
]

__all__ = [
    "VideoDepthAnythingNode",
    "DepthCrafterNode",
    "VIDEO_DEPTH_ESTIMATION_NODES",
]
