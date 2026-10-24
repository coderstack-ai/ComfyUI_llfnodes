import cv2
import numpy as np
import torch
from comfy_api.latest import io

from ...cv_shared import array_to_image, tensor_to_pils

CATEGORY = "llfnodes/图像处理/线段与几何结构检测"


def _detect_lines(gray: np.ndarray):
    if hasattr(cv2, "createLineSegmentDetector"):
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines, *_ = lsd.detect(gray)
        return lines
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createFastLineDetector"):
        return cv2.ximgproc.createFastLineDetector().detect(gray)
    raise RuntimeError("当前 OpenCV 无线段检测器，请安装 opencv-python / opencv-contrib-python。")


def _draw_wireframe(rgb: np.ndarray, lines, draw_on_black: bool, junction_radius: int = 3) -> np.ndarray:
    canvas = np.zeros_like(rgb) if draw_on_black else rgb.copy()
    if lines is None:
        return canvas
    endpoints = []
    for line in lines:
        pts = np.asarray(line).reshape(-1)[:4]
        x0, y0, x1, y1 = [int(round(v)) for v in pts]
        cv2.line(canvas, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
        endpoints.append((x0, y0))
        endpoints.append((x1, y1))
    # cluster nearby endpoints as junctions (HAWP-style wireframe dots)
    if endpoints:
        pts = np.asarray(endpoints, dtype=np.float32)
        used = np.zeros(len(pts), dtype=bool)
        for i, p in enumerate(pts):
            if used[i]:
                continue
            d = np.linalg.norm(pts - p, axis=1)
            group = np.where(d <= 6.0)[0]
            used[group] = True
            c = pts[group].mean(axis=0)
            cv2.circle(canvas, (int(round(c[0])), int(round(c[1]))), junction_radius, (255, 64, 64), -1, cv2.LINE_AA)
    return canvas


class HAWPNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.HAWP",
            display_name="HAWP",
            category=CATEGORY,
            description=(
                "线框检测可视化。官方 hawp 不在 PyPI；本节点用 OpenCV LSD 线段 + 端点结，无需额外安装。"
            ),
            inputs=[
                io.Image.Input("image"),
                io.Boolean.Input("draw_on_black", default=True, label_on="黑底", label_off="原图"),
            ],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image, draw_on_black=True):
        outs = []
        for pil in tensor_to_pils(image):
            rgb = np.asarray(pil.convert("RGB"))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            lines = _detect_lines(gray)
            canvas = _draw_wireframe(rgb, lines, draw_on_black=draw_on_black)
            outs.append(array_to_image(canvas))
        return io.NodeOutput(torch.cat(outs, dim=0))
