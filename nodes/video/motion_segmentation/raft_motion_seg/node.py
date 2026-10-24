import numpy as np
import torch
from comfy_api.latest import io

from ...optical_flow.common import (
    frames_to_pairs,
    get_cached_model,
    get_torch_device,
    pad_to_multiple,
    resize_flow,
)
from ...optical_flow.raft.node import resolve_raft_weight
from ...video_shared import array_batch_to_image, frames_to_video, video_to_frames

CATEGORY = "llfnodes/视频处理/视频运动分割"


class RaftMotionSegNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.RaftMotionSeg",
            display_name="RAFT 运动分割",
            category=CATEGORY,
            description="基于 RAFT 光流幅度的运动分割。权重用 ComfyUI/models/optical_flow/。",
            inputs=[
                io.Video.Input("video"),
                io.Float.Input("mag_threshold", default=1.5, min=0.1, max=50.0, step=0.1),
                io.Int.Input("max_pairs", default=0, min=0, max=512),
            ],
            outputs=[io.Video.Output("VIDEO"), io.Image.Output("MASK")],
        )

    @classmethod
    def execute(cls, video, mag_threshold, max_pairs):
        from torchvision.models.optical_flow import raft_large

        weight_path, weights = resolve_raft_weight("raft_large")

        def loader(device):
            net = raft_large(weights=None, progress=False)
            net.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
            return net.to(device).eval(), weights.transforms()

        net, transforms = get_cached_model(f"raft:motion-seg:{weight_path}", loader)
        device = get_torch_device()
        frames = video_to_frames(video)
        pairs = frames_to_pairs(frames, max_pairs=max_pairs)
        overlays, masks = [], []
        for idx, (img1, img2) in enumerate(pairs):
            _, h, w = img1.shape
            a, _, _ = pad_to_multiple(img1, 8)
            b, _, _ = pad_to_multiple(img2, 8)
            a = a.unsqueeze(0).to(device)
            b = b.unsqueeze(0).to(device)
            a, b = transforms(a, b)
            with torch.inference_mode():
                flow = resize_flow(net(a, b)[-1][0].detach().cpu(), h, w)
            mag = torch.sqrt(flow[0] ** 2 + flow[1] ** 2)
            m = (mag > mag_threshold).float().numpy()
            base = (frames[idx].detach().cpu().numpy() * 255).astype(np.uint8).copy()
            base[m > 0.5] = (base[m > 0.5] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
            overlays.append(base)
            masks.append(m)
        overlays.append(overlays[-1])
        masks.append(masks[-1])
        return io.NodeOutput(
            frames_to_video(array_batch_to_image(overlays), video),
            torch.from_numpy(np.stack(masks).astype(np.float32)),
        )
