from .videomae import VideoMAENode
from .slowfast import SlowFastNode
from .x3d import X3DNode

ACTION_RECOGNITION_NODES = [
    VideoMAENode,
    SlowFastNode,
    X3DNode
]

__all__ = [
    "VideoMAENode",
    "SlowFastNode",
    "X3DNode",
    "ACTION_RECOGNITION_NODES",
]
