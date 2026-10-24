from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import comfy.model_management as model_management
import folder_paths

CATEGORY = "llfnodes/图像处理/前景抠图"

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


def _normalize_mask2d(mask, height: int, width: int) -> np.ndarray:
    """Force mask to float HxW in [0,1], resized to (height, width) if needed."""
    if isinstance(mask, Image.Image):
        mask = np.asarray(mask)
    mask = np.asarray(mask, dtype=np.float32)
    mask = np.squeeze(mask)
    if mask.ndim == 3:
        # HxWxC → take alpha/last channel, or luminance
        mask = mask[..., -1] if mask.shape[-1] in (1, 3, 4) else mask.mean(axis=-1)
    if mask.ndim != 2:
        raise RuntimeError(f"抠图 mask 维度异常: shape={getattr(mask, 'shape', None)}")
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    if mask.max() > 1.5:
        mask = mask / 255.0
    return np.clip(mask, 0.0, 1.0)


def mask_to_outputs(pil: Image.Image, mask: np.ndarray | Image.Image, background: str = "black"):
    """
    mask: HxW float/uint8/PIL
    returns: IMAGE [1,H,W,3], MASK [1,H,W]
    """
    rgb = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    mask = _normalize_mask2d(mask, h, w)
    alpha = mask[..., None]
    if background == "white":
        bg = np.ones_like(rgb)
    elif background == "black":
        bg = np.zeros_like(rgb)
    else:
        bg = rgb
    out = rgb * alpha + bg * (1.0 - alpha)
    image_t = torch.from_numpy(out)[None, ...]
    mask_t = torch.from_numpy(mask)[None, ...]
    return image_t, mask_t


def run_rembg_session(session_name: str, image: torch.Tensor, background: str = "black"):
    from rembg import new_session, remove

    def loader(_device):
        return new_session(session_name)

    session = get_cached_model(f"rembg:{session_name}", loader)

    images = []
    masks = []
    for pil in tensor_to_pils(image):
        rgba = remove(pil, session=session)
        if isinstance(rgba, Image.Image):
            if rgba.mode != "RGBA":
                rgba = rgba.convert("RGBA")
            arr = np.asarray(rgba)
        else:
            arr = np.asarray(rgba)
        if arr.ndim == 3 and arr.shape[-1] >= 4:
            alpha = arr[:, :, 3]
        elif arr.ndim == 2:
            alpha = arr
        else:
            raise RuntimeError(f"rembg 输出异常: shape={arr.shape}")
        # rembg 偶发返回与输入不一致的尺寸，统一按原图对齐
        if alpha.shape[:2] != (pil.height, pil.width):
            alpha = cv2.resize(alpha, (pil.width, pil.height), interpolation=cv2.INTER_LINEAR)
        img_t, mask_t = mask_to_outputs(pil, alpha, background=background)
        images.append(img_t)
        masks.append(mask_t)
    return torch.cat(images, dim=0), torch.cat(masks, dim=0)


def _birefnet_pred_tensor(out) -> torch.Tensor:
    """Normalize BiRefNet / RMBG forward outputs to a mask tensor."""
    # RMBG-1.4: (pred_list, feat_list) → pred_list[-1]
    if isinstance(out, (list, tuple)) and out and isinstance(out[0], (list, tuple)):
        preds = out[0]
        if preds and torch.is_tensor(preds[-1]) and preds[-1].ndim >= 3 and preds[-1].shape[-3] == 1:
            return preds[-1]
    # RMBG-2.0 / BiRefNet: list/tuple of tensors → last
    cur = out
    while isinstance(cur, (list, tuple)) and cur:
        cur = cur[-1]
    if torch.is_tensor(cur):
        return cur
    if hasattr(out, "logits") and torch.is_tensor(out.logits):
        return out.logits
    raise RuntimeError(f"无法解析抠图模型输出类型: {type(out)}")


def run_birefnet_hf(model_id: str, image: torch.Tensor, background: str = "black", image_size: int = 1024):
    from ...model_store import hf_local
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    local = hf_local(model_id)

    def loader(device):
        model = AutoModelForImageSegmentation.from_pretrained(
            local, trust_remote_code=True, local_files_only=True
        )
        model.to(device)
        model.eval()
        return model

    model = get_cached_model(f"birefnet-hf:{local}", loader)
    device = get_torch_device()
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    images = []
    masks = []
    for pil in tensor_to_pils(image):
        inp = transform(pil).unsqueeze(0).to(device)
        with torch.inference_mode():
            preds = _birefnet_pred_tensor(model(inp)).sigmoid().float().cpu()
        pred = preds[0].squeeze()
        mask = transforms.ToPILImage()(pred).resize(pil.size, resample=Image.BILINEAR)
        img_t, mask_t = mask_to_outputs(pil, mask, background=background)
        images.append(img_t)
        masks.append(mask_t)
    return torch.cat(images, dim=0), torch.cat(masks, dim=0)


def modnet_onnx_path() -> str:
    model_dir = Path(folder_paths.models_dir) / "llfnodes" / "modnet"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "model.onnx"
    if not path.exists():
        from ...model_store import ensure_hf_file

        ensure_hf_file("Xenova/modnet", "onnx/model.onnx", path)
    return str(path)


def _modnet_scale(im_h: int, im_w: int, ref_size: int = 512):
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        else:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh, im_rw = im_h, im_w
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    return im_rw / im_w, im_rh / im_h


def run_modnet(image: torch.Tensor, background: str = "black"):
    import onnxruntime as ort

    def loader(device):
        providers = ["CPUExecutionProvider"]
        if device.type == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ort.InferenceSession(modnet_onnx_path(), providers=providers)

    session = get_cached_model("modnet-onnx", loader)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    images = []
    masks = []
    for pil in tensor_to_pils(image):
        im = np.asarray(pil.convert("RGB"), dtype=np.float32)
        im_h, im_w = im.shape[:2]
        x_scale, y_scale = _modnet_scale(im_h, im_w, 512)
        im_norm = (im - 127.5) / 127.5
        im_resized = cv2.resize(im_norm, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_AREA)
        im_input = np.transpose(im_resized, (2, 0, 1))[None, ...].astype(np.float32)
        result = session.run([output_name], {input_name: im_input})[0]
        matte = np.squeeze(result).astype(np.float32)
        matte = cv2.resize(matte, (im_w, im_h), interpolation=cv2.INTER_AREA)
        img_t, mask_t = mask_to_outputs(pil, matte, background=background)
        images.append(img_t)
        masks.append(mask_t)
    return torch.cat(images, dim=0), torch.cat(masks, dim=0)
