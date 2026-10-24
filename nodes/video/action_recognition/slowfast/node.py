import torch
from comfy_api.latest import io

from ....model_store import ensure_torchvision_weights, load_categories_meta
from ...video_shared import get_cached_model, get_torch_device, video_to_frames

CATEGORY = "llfnodes/视频处理/视频动作识别"
WEIGHT_NAME = "slow_fast_r50.pth"


class SlowFastNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.SlowFast",
            display_name="SlowFast",
            category=CATEGORY,
            description="SlowFast 动作识别。权重位于 ComfyUI/models/action_recognition/。",
            inputs=[
                io.Video.Input("video"),
                io.Int.Input("num_frames", default=16, min=4, max=64),
            ],
            outputs=[io.String.Output("PREDICTION")],
        )

    @classmethod
    def execute(cls, video, num_frames):
        from torchvision.models.video import SlowFast_R50_Weights, slow_fast_r50

        weights = SlowFast_R50_Weights.DEFAULT
        weight_path = ensure_torchvision_weights("action_recognition", WEIGHT_NAME, weights)
        categories = load_categories_meta(
            "action_recognition", WEIGHT_NAME, weights.meta.get("categories", [])
        )

        def loader(device):
            model = slow_fast_r50(weights=None)
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            return model.to(device).eval(), weights.transforms(), categories

        model, transforms, cats = get_cached_model(f"slowfast:{weight_path}", loader)
        device = get_torch_device()
        frames = video_to_frames(video)
        t = frames.shape[0]
        idx = torch.linspace(0, t - 1, steps=min(num_frames, t)).long()
        clip = frames[idx].permute(0, 3, 1, 2)
        try:
            inp = transforms(clip)
        except Exception:
            inp = clip
        if inp.ndim == 4:
            inp = inp.permute(1, 0, 2, 3).unsqueeze(0)
        elif inp.ndim != 5:
            inp = inp.unsqueeze(0)
        inp = inp.to(device)
        with torch.inference_mode():
            logits = model(inp)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            prob = torch.softmax(logits, dim=-1)[0]
            topk = torch.topk(prob, k=min(5, prob.numel()))
        lines = []
        for score, i in zip(topk.values.tolist(), topk.indices.tolist()):
            name = cats[i] if i < len(cats) else str(i)
            lines.append(f"{name}: {score:.4f}")
        return io.NodeOutput("\n".join(lines))
