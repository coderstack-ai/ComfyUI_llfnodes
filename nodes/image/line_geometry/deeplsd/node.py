import cv2
import numpy as np
import torch
from comfy_api.latest import io
from pathlib import Path

from ...cv_shared import array_to_image, get_torch_device, tensor_to_pils

CATEGORY = "llfnodes/图像处理/线段与几何结构检测"


def _run_deeplsd(gray: np.ndarray, device: torch.device):
    try:
        from deeplsd.models.deeplsd_inference import DeepLSD
    except ImportError as e:
        raise RuntimeError(
            "DeepLSD 未安装。请按 https://github.com/cvg/DeepLSD 安装，"
            "并将权重放到 ComfyUI/models/llfnodes/deeplsd/。"
        ) from e

    import folder_paths

    weight_dir = Path(folder_paths.models_dir) / "llfnodes" / "deeplsd"
    ckpt = next(
        (p for p in (weight_dir / "deeplsd_md.tar", weight_dir / "deeplsd_wireframe.tar") if p.is_file()),
        None,
    )
    if ckpt is None:
        raise RuntimeError(
            f"未找到 DeepLSD 权重。请将 deeplsd_md.tar 或 deeplsd_wireframe.tar 放到 {weight_dir}"
        )

    conf = {
        "detect_lines": True,
        "line_detection_params": {
            "merge": False,
            "filtering": True,
            "grad_thresh": 3,
            "grad_nfa": True,
        },
    }
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
    lines = np.asarray(lines)
    if lines.size == 0:
        return np.zeros((0, 1, 4), dtype=np.float32)
    return lines.reshape(-1, lines.shape[-1])[:, :4].reshape(-1, 1, 4)


class DeepLSDNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DeepLSD",
            display_name="DeepLSD",
            category=CATEGORY,
            description="DeepLSD 线段检测。需官方包与权重；失败直接报错。",
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
            lines = _run_deeplsd(gray, device)
            canvas = np.zeros_like(rgb) if draw_on_black else rgb.copy()
            for line in lines:
                x0, y0, x1, y1 = [int(round(v)) for v in np.asarray(line).reshape(-1)[:4]]
                cv2.line(canvas, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
            outs.append(array_to_image(canvas))
        return io.NodeOutput(torch.cat(outs, dim=0))
