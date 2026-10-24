import torch
from comfy_api.latest import io

from ..common import (
    CATEGORY,
    flow_to_rgb,
    frames_to_pairs,
    get_cached_model,
    get_torch_device,
    pad_to_multiple,
    resize_flow,
    run_raft_fallback,
    video_to_frames,
)


class SEARAFTNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.SEARAFT",
            display_name="SEA-RAFT",
            category=CATEGORY,
            description="SEA-RAFT 光流。未安装官方依赖/权重时回退到 torchvision RAFT。",
            inputs=[
                io.Video.Input("video"),
                io.String.Input("model_path", default="", tooltip="SEA-RAFT 权重路径（.pth）；空则 RAFT 回退"),
                io.Int.Input("iters", default=4, min=1, max=32),
                io.Int.Input("max_pairs", default=0, min=0, max=512),
            ],
            outputs=[io.Image.Output("FLOW")],
        )

    @classmethod
    def execute(cls, video, model_path, iters, max_pairs):
        model_cls = None
        for mod_name, attr in [
            ("sea_raft", "RAFT"),
            ("searraft", "RAFT"),
            ("core.raft", "RAFT"),
        ]:
            try:
                mod = __import__(mod_name, fromlist=[attr])
                model_cls = getattr(mod, attr)
                break
            except Exception:
                pass

        if model_cls is None or not model_path:
            return io.NodeOutput(run_raft_fallback(video, max_pairs=max_pairs))

        def loader(device):
            try:
                model = model_cls({"dropout": 0, "mixed_precision": False, "alternate_corr": False})
            except TypeError:
                model = model_cls()
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            if isinstance(state, dict) and any(k.startswith("module.") for k in state):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            return model.to(device).eval()

        model = get_cached_model(f"searraft:{model_path}", loader)
        device = get_torch_device()
        pairs = frames_to_pairs(video_to_frames(video), max_pairs=max_pairs)
        outs = []
        for img1, img2 in pairs:
            _, h, w = img1.shape
            a, _, _ = pad_to_multiple(img1, 8)
            b, _, _ = pad_to_multiple(img2, 8)
            a = a.unsqueeze(0).to(device)
            b = b.unsqueeze(0).to(device)
            with torch.inference_mode():
                try:
                    flow = model(a, b, iters=iters)
                except TypeError:
                    flow = model(a, b)
                if isinstance(flow, (list, tuple)):
                    flow = flow[-1]
                flow = flow[0].detach().cpu()
            outs.append(flow_to_rgb(resize_flow(flow, h, w)))
        return io.NodeOutput(torch.cat(outs, dim=0))
