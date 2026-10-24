import torch
from comfy_api.latest import io
from torchvision.transforms.functional import to_pil_image

from ....model_store import ensure_hf_repo
from ...video_shared import get_cached_model, get_torch_device, video_to_frames

CATEGORY = "llfnodes/视频处理/视频动作识别"
MODELS = {
    "base": ("videomae-base", "MCG-NJU/videomae-base-finetuned-kinetics"),
    "large": ("videomae-large", "MCG-NJU/videomae-large-finetuned-kinetics"),
}


class VideoMAENode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.VideoMAE",
            display_name="VideoMAE",
            category=CATEGORY,
            description="VideoMAE。权重位于 ComfyUI/models/video_mae/，缺失时自动下载。",
            inputs=[
                io.Video.Input("video"),
                io.Combo.Input("model", options=list(MODELS), default="base"),
                io.Int.Input("num_frames", default=16, min=8, max=32),
            ],
            outputs=[io.String.Output("PREDICTION")],
        )

    @classmethod
    def execute(cls, video, model, num_frames):
        from transformers import AutoImageProcessor, AutoModelForVideoClassification

        dirname, repo_id = MODELS[model]
        local_dir = ensure_hf_repo("video_mae", dirname, repo_id)

        def loader(device):
            return (
                AutoImageProcessor.from_pretrained(str(local_dir), local_files_only=True),
                AutoModelForVideoClassification.from_pretrained(str(local_dir), local_files_only=True)
                .to(device)
                .eval(),
            )

        processor, net = get_cached_model(f"videomae:{local_dir}", loader)
        device = get_torch_device()
        frames = video_to_frames(video)
        t = frames.shape[0]
        idx = torch.linspace(0, t - 1, steps=min(num_frames, t)).long()
        pils = [to_pil_image(frames[i].permute(2, 0, 1).cpu()) for i in idx.tolist()]
        inputs = {k: v.to(device) for k, v in processor(pils, return_tensors="pt").items()}
        with torch.inference_mode():
            logits = net(**inputs).logits[0]
            prob = torch.softmax(logits, dim=-1)
            topk = torch.topk(prob, k=min(5, prob.numel()))
        id2label = net.config.id2label
        lines = [f"{id2label.get(i, i)}: {s:.4f}" for s, i in zip(topk.values.tolist(), topk.indices.tolist())]
        return io.NodeOutput("\n".join(lines))
