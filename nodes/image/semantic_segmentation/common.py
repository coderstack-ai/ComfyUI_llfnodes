import numpy as np
import torch
from PIL import Image

import comfy.model_management as model_management

CATEGORY = "llfnodes/图像处理/语义分割"

_model_cache: dict[str, tuple] = {}

# ADE20K-style palette (150+); cycles if more classes
_PALETTE = np.array(
    [
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [119, 11, 32],
        [2, 255, 130],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [128, 0, 0],
        [0, 128, 0],
        [0, 0, 128],
        [128, 128, 0],
        [128, 0, 128],
        [0, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ],
    dtype=np.uint8,
)


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


def label_map_to_color(label_map: np.ndarray) -> np.ndarray:
    label_map = np.asarray(label_map, dtype=np.int64)
    h, w = label_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in np.unique(label_map):
        color[label_map == cls_id] = _PALETTE[int(cls_id) % len(_PALETTE)]
    return color


def colorize_and_blend(pil: Image.Image, label_map: np.ndarray, alpha: float) -> torch.Tensor:
    color = label_map_to_color(label_map).astype(np.float32) / 255.0
    orig = np.asarray(pil, dtype=np.float32) / 255.0
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = orig * (1.0 - alpha) + color * alpha
    return torch.from_numpy(blended)[None, ...]


def label_map_to_mask_image(label_map: np.ndarray) -> torch.Tensor:
    """Class-id map as single-channel grayscale IMAGE (values scaled to 0-1)."""
    label_map = np.asarray(label_map, dtype=np.float32)
    max_id = max(float(label_map.max()), 1.0)
    gray = (label_map / max_id)[..., None]
    return torch.from_numpy(np.repeat(gray, 3, axis=-1))[None, ...]


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


def run_semantic_auto_model(model_id: str, image: torch.Tensor, alpha: float = 0.6):
    """SegFormer / UPerNet style AutoModelForSemanticSegmentation."""
    from ...model_store import hf_local
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    local = hf_local(model_id)

    def loader(device):
        processor = AutoImageProcessor.from_pretrained(local, local_files_only=True)
        model = AutoModelForSemanticSegmentation.from_pretrained(local, local_files_only=True)
        model.to(device)
        model.eval()
        return processor, model

    processor, model = get_cached_model(f"semseg:{local}", loader)
    device = get_torch_device()

    overlays = []
    maps = []
    for pil in tensor_to_pils(image):
        inputs = processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=pil.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        label_map = upsampled.argmax(dim=1)[0].detach().cpu().numpy()
        overlays.append(colorize_and_blend(pil, label_map, alpha))
        maps.append(label_map_to_mask_image(label_map))

    return torch.cat(overlays, dim=0), torch.cat(maps, dim=0)


def run_mask2former(model_id: str, image: torch.Tensor, alpha: float = 0.6):
    from ...model_store import hf_local
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    local = hf_local(model_id)

    def loader(device):
        processor = AutoImageProcessor.from_pretrained(local, local_files_only=True)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(local, local_files_only=True)
        model.to(device)
        model.eval()
        return processor, model

    processor, model = get_cached_model(f"mask2former:{local}", loader)
    device = get_torch_device()

    overlays = []
    maps = []
    for pil in tensor_to_pils(image):
        inputs = processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        result = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[pil.size[::-1]]
        )[0]
        label_map = result.detach().cpu().numpy()
        overlays.append(colorize_and_blend(pil, label_map, alpha))
        maps.append(label_map_to_mask_image(label_map))

    return torch.cat(overlays, dim=0), torch.cat(maps, dim=0)


def run_oneformer(model_id: str, image: torch.Tensor, alpha: float = 0.6):
    from ...model_store import hf_local
    from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

    local = hf_local(model_id)

    def loader(device):
        processor = OneFormerProcessor.from_pretrained(local, local_files_only=True)
        model = OneFormerForUniversalSegmentation.from_pretrained(local, local_files_only=True)
        model.to(device)
        model.eval()
        return processor, model

    processor, model = get_cached_model(f"oneformer:{local}", loader)
    device = get_torch_device()

    overlays = []
    maps = []
    for pil in tensor_to_pils(image):
        inputs = processor(images=pil, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        result = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[pil.size[::-1]]
        )[0]
        label_map = result.detach().cpu().numpy()
        overlays.append(colorize_and_blend(pil, label_map, alpha))
        maps.append(label_map_to_mask_image(label_map))

    return torch.cat(overlays, dim=0), torch.cat(maps, dim=0)
