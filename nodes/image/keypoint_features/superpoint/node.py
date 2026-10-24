import cv2
import numpy as np
import torch
from comfy_api.latest import io

from ...cv_shared import array_to_image, get_cached_model, get_torch_device, require_pkg, tensor_to_pils

CATEGORY = "llfnodes/图像处理/图像关键点与特征提取"


class SuperPointNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.SuperPoint",
            display_name="SuperPoint",
            category=CATEGORY,
            description="SuperPoint 关键点（需 kornia 提供 KF.SuperPoint；kornia 0.8+ 已移除则直接报错）。",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("max_keypoints", default=1024, min=16, max=8192),
            ],
            outputs=[io.Image.Output("IMAGE"), io.String.Output("KEYPOINTS")],
        )

    @classmethod
    def execute(cls, image, max_keypoints):
        require_pkg("kornia", "pip install kornia")
        import kornia.feature as KF
        from torchvision.transforms.functional import to_tensor

        if not hasattr(KF, "SuperPoint"):
            raise RuntimeError(
                "当前 kornia 无 SuperPoint（0.8+ 已移除）。"
                "请改用 DISK / ALIKED 节点，或安装仍包含 SuperPoint 的 kornia 版本。"
            )

        def loader(device):
            return KF.SuperPoint(num_filters=64, max_num_keypoints=max_keypoints).to(device).eval()

        model = get_cached_model("superpoint", loader)
        device = get_torch_device()
        outs, texts = [], []
        for pil in tensor_to_pils(image):
            img = to_tensor(pil.convert("L")).unsqueeze(0).to(device)
            with torch.inference_mode():
                out = model({"image": img})
            kpts = out["keypoints"][0].detach().cpu().numpy()[:max_keypoints]
            canvas = np.asarray(pil.convert("RGB")).copy()
            for x, y in kpts:
                cv2.circle(canvas, (int(x), int(y)), 2, (0, 255, 0), -1)
            outs.append(array_to_image(canvas))
            texts.append("\n".join(f"{x:.1f},{y:.1f}" for x, y in kpts[:200]))
        return io.NodeOutput(torch.cat(outs, dim=0), "\n---\n".join(texts))
