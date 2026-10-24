from .mmpose import MMPoseTrackNode
from .posetrack import PoseTrackNode

POSE_TRACKING_NODES = [
    MMPoseTrackNode,
    PoseTrackNode
]

__all__ = [
    "MMPoseTrackNode",
    "PoseTrackNode",
    "POSE_TRACKING_NODES",
]
