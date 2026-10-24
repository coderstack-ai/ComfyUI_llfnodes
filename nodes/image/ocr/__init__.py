from .paddleocr import PaddleOCRNode
from .trocr import TrOCRNode
from .dbnet import DBNetNode

OCR_NODES = [
    PaddleOCRNode,
    TrOCRNode,
    DBNetNode
]

__all__ = [
    "PaddleOCRNode",
    "TrOCRNode",
    "DBNetNode",
    "OCR_NODES",
]
