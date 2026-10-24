import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import comfy.model_management as model_management
import folder_paths

CATEGORY = "llfnodes/图像处理/人体姿态估计"

_model_cache: dict[str, object] = {}


def get_torch_device():
    return model_management.get_torch_device()


def tensor_to_pils(image: torch.Tensor) -> list[Image.Image]:
    imgs = []
    for i in range(image.shape[0]):
        arr = (image[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        imgs.append(Image.fromarray(arr, mode="RGB"))
    return imgs


def array_to_image(arr: np.ndarray | torch.Tensor | Image.Image) -> torch.Tensor:
    if isinstance(arr, Image.Image):
        arr = np.asarray(arr)
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().float().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return torch.from_numpy(np.clip(arr, 0.0, 1.0))[None, ...]


def get_cached_model(cache_key: str, loader):
    device = get_torch_device()
    key = f"{cache_key}@{device}"
    cached = _model_cache.get(key)
    if cached is not None:
        return cached
    model_management.unload_all_models()
    value = loader(device)
    _model_cache.clear()
    _model_cache[key] = value
    return value


def ensure_controlnet_aux():
    try:
        import custom_controlnet_aux  # noqa: F401
        return
    except ImportError:
        pass

    custom_nodes = Path(__file__).resolve().parents[4]
    aux_src = custom_nodes / "comfyui_controlnet_aux" / "src"
    if aux_src.is_dir() and str(aux_src) not in sys.path:
        sys.path.insert(0, str(aux_src))
    try:
        import custom_controlnet_aux  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "需要安装 comfyui_controlnet_aux（提供 DWPose / OpenPose）。\n"
            "请用 ComfyUI Manager 安装 ControlNet Auxiliary Preprocessors。"
        ) from e


def run_controlnet_pose_detector(detector, image: torch.Tensor, **kwargs) -> torch.Tensor:
    outs = []
    for pil in tensor_to_pils(image):
        result = detector(np.asarray(pil), output_type="np", **kwargs)
        if isinstance(result, tuple):
            result = result[0]
        outs.append(array_to_image(result))
    return torch.cat(outs, dim=0)


def device_str_for_onnx() -> str:
    device = get_torch_device()
    if device.type == "cuda":
        return "cuda"
    return "cpu"


def mediapipe_model_path(variant: str = "lite") -> str:
    """Download MediaPipe pose landmarker .task into ComfyUI models/llfnodes if needed."""
    names = {
        "lite": "pose_landmarker_lite.task",
        "full": "pose_landmarker_full.task",
        "heavy": "pose_landmarker_heavy.task",
    }
    urls = {
        "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    }
    filename = names[variant]
    model_dir = Path(folder_paths.models_dir) / "llfnodes" / "mediapipe"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / filename
    if not path.exists():
        from ...model_store import _download_url

        _download_url(urls[variant], path)
    return str(path)


def draw_mediapipe_pose(rgb: np.ndarray, landmarks_list, min_score: float = 0.5) -> np.ndarray:
    from mediapipe.tasks.python.vision import PoseLandmarksConnections

    h, w = rgb.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for landmarks in landmarks_list:
        pts = []
        for lm in landmarks:
            vis = getattr(lm, "visibility", 1.0) or 1.0
            presence = getattr(lm, "presence", 1.0) or 1.0
            score = min(float(vis), float(presence))
            pts.append((int(lm.x * w), int(lm.y * h), score))

        for conn in PoseLandmarksConnections.POSE_LANDMARKS:
            a, b = pts[conn.start], pts[conn.end]
            if a[2] >= min_score and b[2] >= min_score:
                cv2.line(canvas, (a[0], a[1]), (b[0], b[1]), (0, 255, 0), 2, cv2.LINE_AA)
        for x, y, score in pts:
            if score >= min_score:
                cv2.circle(canvas, (x, y), 3, (255, 0, 0), -1, cv2.LINE_AA)
    return canvas
