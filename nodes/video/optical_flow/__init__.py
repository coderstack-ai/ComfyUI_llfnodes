from .raft import RAFTNode
from .gmflow import GMFlowNode
from .sea_raft import SEARAFTNode
from .flowformer import FlowFormerNode

OPTICAL_FLOW_NODES = [
    RAFTNode,
    GMFlowNode,
    SEARAFTNode,
    FlowFormerNode,
]

__all__ = [
    "RAFTNode",
    "GMFlowNode",
    "SEARAFTNode",
    "FlowFormerNode",
    "OPTICAL_FLOW_NODES",
]
