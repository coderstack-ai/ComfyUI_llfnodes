import numpy as np
import torch
from PIL import Image

import comfy.model_management as model_management

CATEGORY = "llfnodes/图像处理/实例分割"

_model_cache: dict[str, object] = {}

_PALETTE = np.array(
    [
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
        [255, 128, 0],
        [255, 0, 128],
        [128, 255, 0],
        [0, 255, 128],
        [128, 0, 255],
        [0, 128, 255],
        [255, 128, 128],
        [128, 255, 128],
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


def instance_masks_to_outputs(
    pil: Image.Image,
    masks: list[np.ndarray],
    alpha: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    masks: list of bool/float HxW
    returns: overlay IMAGE [1,H,W,3], combined MASK [1,H,W]
    """
    h, w = pil.height, pil.width
    color = np.zeros((h, w, 3), dtype=np.float32)
    combined = np.zeros((h, w), dtype=np.float32)

    for idx, mask in enumerate(masks):
        m = np.asarray(mask)
        if m.ndim > 2:
            m = np.squeeze(m)
        if m.shape != (h, w):
            m_t = torch.from_numpy(m.astype(np.float32))[None, None]
            m_t = torch.nn.functional.interpolate(m_t, size=(h, w), mode="nearest")
            m = m_t.squeeze().numpy()
        bin_m = m > 0.5
        if not bin_m.any():
            continue
        rgb = _PALETTE[idx % len(_PALETTE)].astype(np.float32) / 255.0
        color[bin_m] = rgb
        combined = np.maximum(combined, bin_m.astype(np.float32))

    orig = np.asarray(pil, dtype=np.float32) / 255.0
    a = float(np.clip(alpha, 0.0, 1.0))
    has = combined > 0
    blended = orig.copy()
    blended[has] = orig[has] * (1.0 - a) + color[has] * a
    overlay = torch.from_numpy(blended)[None, ...]
    mask_out = torch.from_numpy(combined)[None, ...]
    return overlay, mask_out


def label_map_to_instance_outputs(pil: Image.Image, label_map: np.ndarray, alpha: float = 0.6):
    label_map = np.asarray(label_map)
    masks = []
    for inst_id in np.unique(label_map):
        if int(inst_id) <= 0:
            continue
        masks.append(label_map == inst_id)
    return instance_masks_to_outputs(pil, masks, alpha=alpha)


def run_mask_generation(model_id: str, image: torch.Tensor, points_per_batch: int = 64, alpha: float = 0.6):
    from transformers import pipeline

    def loader(device):
        device_idx = device.index if device.type == "cuda" and device.index is not None else (0 if device.type == "cuda" else -1)
        return pipeline("mask-generation", model=model_id, device=device_idx)

    pipe = get_cached_model(f"maskgen:{model_id}", loader)

    overlays = []
    masks_out = []
    for pil in tensor_to_pils(image):
        result = pipe(pil, points_per_batch=points_per_batch)
        raw_masks = result["masks"] if isinstance(result, dict) else result
        np_masks = []
        for m in raw_masks:
            if isinstance(m, Image.Image):
                np_masks.append(np.array(m) > 0)
            elif isinstance(m, torch.Tensor):
                np_masks.append(m.detach().cpu().numpy() > 0.5)
            else:
                np_masks.append(np.asarray(m) > 0.5)
        ov, mk = instance_masks_to_outputs(pil, np_masks, alpha=alpha)
        overlays.append(ov)
        masks_out.append(mk)

    return torch.cat(overlays, dim=0), torch.cat(masks_out, dim=0)


def run_mask2former_instance(model_id: str, image: torch.Tensor, threshold: float = 0.5, alpha: float = 0.6):
    from ...model_store import hf_local
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    local = hf_local(model_id)

    def loader(device):
        processor = AutoImageProcessor.from_pretrained(local, local_files_only=True)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(local, local_files_only=True)
        model.to(device)
        model.eval()
        return processor, model

    processor, model = get_cached_model(f"m2f-inst:{local}", loader)
    device = get_torch_device()

    overlays = []
    masks_out = []
    for pil in tensor_to_pils(image):
        inputs = processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[pil.size[::-1]],
            threshold=threshold,
        )[0]
        seg = result["segmentation"]
        if isinstance(seg, torch.Tensor):
            seg = seg.detach().cpu().numpy()
        ov, mk = label_map_to_instance_outputs(pil, seg, alpha=alpha)
        overlays.append(ov)
        masks_out.append(mk)

    return torch.cat(overlays, dim=0), torch.cat(masks_out, dim=0)


def run_mask_rcnn(image: torch.Tensor, score_threshold: float = 0.5, alpha: float = 0.6):
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
    from torchvision.transforms.functional import to_tensor

    def loader(device):
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn_v2(weights=weights)
        model.to(device)
        model.eval()
        return model, weights

    model, weights = get_cached_model("maskrcnn:resnet50_v2", loader)
    device = get_torch_device()

    overlays = []
    masks_out = []
    for pil in tensor_to_pils(image):
        img_t = to_tensor(pil).to(device)
        pred = model([img_t])[0]
        keep = pred["scores"] >= score_threshold
        masks = pred["masks"][keep, 0].detach().cpu().numpy()
        np_masks = [m > 0.5 for m in masks]
        ov, mk = instance_masks_to_outputs(pil, np_masks, alpha=alpha)
        overlays.append(ov)
        masks_out.append(mk)

    return torch.cat(overlays, dim=0), torch.cat(masks_out, dim=0)
