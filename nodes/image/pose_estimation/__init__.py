from .dwpose import DWPoseNode
from .openpose import OpenPoseNode
from .rtmpose import RTMPoseNode
from .mediapipe_pose import MediaPipePoseNode

POSE_ESTIMATION_NODES = [
    DWPoseNode,
    OpenPoseNode,
    RTMPoseNode,
    MediaPipePoseNode,
]

__all__ = [
    "DWPoseNode",
    "OpenPoseNode",
    "RTMPoseNode",
    "MediaPipePoseNode",
    "POSE_ESTIMATION_NODES",
]
