import torch
from comfy_api.latest import io

from ....model_store import ensure_torchvision_weights
from ..common import (
    CATEGORY,
    flow_to_rgb,
    frames_to_pairs,
    get_cached_model,
    get_torch_device,
    pad_to_multiple,
    resize_flow,
    video_to_frames,
)


def resolve_raft_weight(model: str):
    from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights

    if model == "raft_small":
        weights = Raft_Small_Weights.DEFAULT
        name = "raft_small.pth"
    else:
        weights = Raft_Large_Weights.DEFAULT
        name = "raft_large.pth"
    path = ensure_torchvision_weights("optical_flow", name, weights)
    return path, weights


class RAFTNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.RAFT",
            display_name="RAFT",
            category=CATEGORY,
            description="RAFT 光流。权重位于 ComfyUI/models/optical_flow/，缺失/损坏时自动下载。",
            inputs=[
                io.Video.Input("video"),
                io.Combo.Input("model", options=["raft_large", "raft_small"], default="raft_large"),
                io.Int.Input("max_pairs", default=0, min=0, max=512, tooltip="0 表示处理全部相邻帧对"),
            ],
            outputs=[io.Image.Output("FLOW")],
        )

    @classmethod
    def execute(cls, video, model, max_pairs):
        from torchvision.models.optical_flow import raft_large, raft_small

        weight_path, weights = resolve_raft_weight(model)

        def loader(device):
            net = raft_small(weights=None, progress=False) if model == "raft_small" else raft_large(weights=None, progress=False)
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
        return io.NodeOutput(torch.cat(outs, dim=0))
