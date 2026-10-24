import cv2
import numpy as np
import torch
from comfy_api.latest import io

from ...cv_shared import array_to_image, get_torch_device, tensor_to_pils

CATEGORY = "llfnodes/图像处理/线段与几何结构检测"


def _detect_lines_opencv(gray: np.ndarray):
    """OpenCV LSD (PyPI 无 deeplsd；官方 DeepLSD 需从 GitHub 编译安装)."""
    if hasattr(cv2, "createLineSegmentDetector"):
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines, *_ = lsd.detect(gray)
        return lines
    # older opencv-contrib
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createFastLineDetector"):
        fld = cv2.ximgproc.createFastLineDetector()
        return fld.detect(gray)
    raise RuntimeError("当前 OpenCV 无线段检测器，请安装 opencv-python / opencv-contrib-python。")


def _try_deeplsd(gray: np.ndarray, device: torch.device):
    """Optional real DeepLSD if cvg/DeepLSD + weights are installed."""
    try:
        from deeplsd.models.deeplsd_inference import DeepLSD
    except ImportError:
        return None

    import folder_paths
    from pathlib import Path

    weight_dir = Path(folder_paths.models_dir) / "llfnodes" / "deeplsd"
    ckpt = next((p for p in (weight_dir / "deeplsd_md.tar", weight_dir / "deeplsd_wireframe.tar") if p.is_file()), None)
    if ckpt is None:
        return None

    conf = {
        "detect_lines": True,
        "line_detection_params": {
            "merge": False,
            "filtering": True,
            "grad_thresh": 3,
            "grad_nfa": True,
        },
    }
    try:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        net = DeepLSD(conf)
        net.load_state_dict(state["model"])
        net = net.to(device).eval()
        inputs = {"image": torch.from_numpy(gray)[None, None].float().to(device) / 255.0}
        with torch.inference_mode():
            out = net(inputs)
        lines = out["lines"][0] if isinstance(out, dict) else out
        if torch.is_tensor(lines):
            lines = lines.detach().cpu().numpy()
        lines = np.asarray(lines).reshape(-1, lines.shape[-1] if hasattr(lines, "shape") else 4)
        if lines.size == 0:
            return np.zeros((0, 1, 4), dtype=np.float32)
        return lines[:, :4].reshape(-1, 1, 4)
    except Exception:
        return None


class DeepLSDNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DeepLSD",
            display_name="DeepLSD",
            category=CATEGORY,
            description=(
                "线段检测。优先用已安装的 DeepLSD（权重放 models/llfnodes/deeplsd/）；"
                "否则回退 OpenCV LSD（无需额外 pip）。"
            ),
            inputs=[
                io.Image.Input("image"),
                io.Boolean.Input("draw_on_black", default=True, label_on="黑底", label_off="原图"),
            ],
            outputs=[io.Image.Output("IMAGE")],
        )

    @classmethod
    def execute(cls, image, draw_on_black=True):
        device = get_torch_device()
        outs = []
        for pil in tensor_to_pils(image):
            rgb = np.asarray(pil.convert("RGB"))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            lines = _try_deeplsd(gray, device)
            if lines is None:
                lines = _detect_lines_opencv(gray)
            canvas = np.zeros_like(rgb) if draw_on_black else rgb.copy()
            if lines is not None:
                for line in lines:
                    x0, y0, x1, y1 = [int(round(v)) for v in np.asarray(line).reshape(-1)[:4]]
                    cv2.line(canvas, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
            outs.append(array_to_image(canvas))
        return io.NodeOutput(torch.cat(outs, dim=0))
