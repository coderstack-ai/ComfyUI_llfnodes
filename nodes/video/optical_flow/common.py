import numpy as np
import torch

import comfy.model_management as model_management

CATEGORY = "llfnodes/视频处理/光流估计"

_model_cache: dict[str, object] = {}


def get_torch_device():
    return model_management.get_torch_device()


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


def require_pkg(pkg: str, install_hint: str):
    try:
        return __import__(pkg)
    except ImportError as e:
        raise RuntimeError(f"缺少依赖 {pkg}。\n安装：{install_hint}") from e


def video_to_frames(video) -> torch.Tensor:
    """Return frames as float IMAGE tensor [T,H,W,C] in [0,1]."""
    components = video.get_components()
    frames = components.images
    if frames.dtype != torch.float32:
        frames = frames.float()
        if frames.max() > 1.5:
            frames = frames / 255.0
    return frames


def frames_to_pairs(frames: torch.Tensor, max_pairs: int = 0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """frames: [T,H,W,C] -> consecutive pairs as CHW float tensors."""
    if frames.ndim != 4 or frames.shape[0] < 2:
        raise RuntimeError("光流估计至少需要 2 帧（视频或图像 batch）。")
    t = frames.shape[0]
    n = t - 1
    if max_pairs > 0:
        n = min(n, max_pairs)
    pairs = []
    for i in range(n):
        a = frames[i].permute(2, 0, 1).contiguous()
        b = frames[i + 1].permute(2, 0, 1).contiguous()
        pairs.append((a, b))
    return pairs


def flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
    """flow: (2,H,W) -> IMAGE (1,H,W,3) in [0,1]."""
    try:
        from torchvision.utils import flow_to_image

        rgb = flow_to_image(flow.detach().float().cpu())  # (3,H,W) uint8
        return (rgb.float() / 255.0).permute(1, 2, 0).unsqueeze(0)
    except Exception:
        # HSV wheel fallback
        flow = flow.detach().float().cpu().numpy()
        u, v = flow[0], flow[1]
        mag = np.sqrt(u * u + v * v)
        ang = np.arctan2(v, u)
        hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.float32)
        hsv[..., 0] = (ang + np.pi) / (2 * np.pi)
        hsv[..., 1] = 1.0
        hsv[..., 2] = np.clip(mag / (np.percentile(mag, 99) + 1e-6), 0, 1)
        import cv2

        rgb = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb)[None, ...]


def pad_to_multiple(img: torch.Tensor, multiple: int = 8):
    """img CHW -> padded CHW, returns (padded, orig_h, orig_w)."""
    _, h, w = img.shape
    nh = ((h + multiple - 1) // multiple) * multiple
    nw = ((w + multiple - 1) // multiple) * multiple
    if nh == h and nw == w:
        return img, h, w
    out = torch.nn.functional.pad(img, (0, nw - w, 0, nh - h), mode="replicate")
    return out, h, w


def resize_flow(flow: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if flow.shape[-2:] == (h, w):
        return flow
    oh, ow = flow.shape[-2:]
    flow = torch.nn.functional.interpolate(flow[None], size=(h, w), mode="bilinear", align_corners=False)[0]
    flow[0] *= w / max(ow, 1)
    flow[1] *= h / max(oh, 1)
    return flow


def run_raft_fallback(video, max_pairs: int = 0, model: str = "raft_large"):
    """Shared RAFT path used when GMFlow / FlowFormer / SEA-RAFT backends are unavailable."""
    from torchvision.models.optical_flow import raft_large, raft_small

    from ...model_store import ensure_torchvision_weights

    if model == "raft_small":
        from torchvision.models.optical_flow import Raft_Small_Weights as W

        name = "raft_small.pth"
        factory = raft_small
    else:
        from torchvision.models.optical_flow import Raft_Large_Weights as W

        name = "raft_large.pth"
        factory = raft_large
    weights = W.DEFAULT
    weight_path = ensure_torchvision_weights("optical_flow", name, weights)

    def loader(device):
        net = factory(weights=None, progress=False)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        return net.to(device).eval(), weights.transforms()

    net, transforms = get_cached_model(f"raft:{weight_path}", loader)
    device = get_torch_device()
    pairs = frames_to_pairs(video_to_frames(video), max_pairs=max_pairs)
    outs = []
    for img1, img2 in pairs:
        _, h, w = img1.shape
        a, _, _ = pad_to_multiple(img1, 8)
        b, _, _ = pad_to_multiple(img2, 8)
        a = a.unsqueeze(0).to(device)
        b = b.unsqueeze(0).to(device)
        a, b = transforms(a, b)
        with torch.inference_mode():
            flows = net(a, b)
        flow = resize_flow(flows[-1][0].detach().cpu(), h, w)
        outs.append(flow_to_rgb(flow))
    return torch.cat(outs, dim=0)
