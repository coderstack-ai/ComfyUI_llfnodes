import torch
from comfy_api.latest import io

from ..common import (
    CATEGORY,
    flow_to_rgb,
    frames_to_pairs,
    get_cached_model,
    get_torch_device,
    pad_to_multiple,
    require_pkg,
    resize_flow,
    video_to_frames,
)


class GMFlowNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.GMFlow",
            display_name="GMFlow",
            category=CATEGORY,
            description="GMFlow 光流。需安装 gmflow 并填写权重路径；失败直接报错。",
            inputs=[
                io.Video.Input("video"),
                io.String.Input("model_path", default="", tooltip="GMFlow 权重路径（.pth）"),
                io.Int.Input("max_pairs", default=0, min=0, max=512),
            ],
            outputs=[io.Image.Output("FLOW")],
        )

    @classmethod
    def execute(cls, video, model_path, max_pairs):
        require_pkg("gmflow", "pip install gmflow 或按 https://github.com/haofeixu/gmflow 安装")
        if not model_path:
            raise RuntimeError("GMFlow 需要填写 model_path（权重 .pth）。")

        try:
            from gmflow.gmflow import GMFlow
        except Exception as e:
            raise RuntimeError(
                "已检测到 gmflow 包名，但无法导入 GMFlow 模型类。请按官方仓库安装完整代码与权重。"
            ) from e

        def loader(device):
            model = GMFlow(
                feature_channels=128,
                num_scales=1,
                upsample_factor=8,
                num_head=1,
                attention_type="swin",
                ffn_dim_expansion=4,
                num_transformer_layers=6,
            ).to(device)
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model.load_state_dict(state, strict=False)
            return model.eval()

        model = get_cached_model(f"gmflow:{model_path}", loader)
        device = get_torch_device()
        pairs = frames_to_pairs(video_to_frames(video), max_pairs=max_pairs)
        outs = []
        for img1, img2 in pairs:
            _, h, w = img1.shape
            a, _, _ = pad_to_multiple(img1, 8)
            b, _, _ = pad_to_multiple(img2, 8)
            a = (a.unsqueeze(0) * 255.0).to(device)
            b = (b.unsqueeze(0) * 255.0).to(device)
            with torch.inference_mode():
                results = model(
                    a,
                    b,
                    attn_splits_list=[2],
                    corr_radius_list=[-1],
                    prop_radius_list=[-1],
                )
                flow = results["flow_preds"][-1][0].detach().cpu()
            outs.append(flow_to_rgb(resize_flow(flow, h, w)))
        return io.NodeOutput(torch.cat(outs, dim=0))
