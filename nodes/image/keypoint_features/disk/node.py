from comfy_api.latest import io
from ...cv_shared import array_to_image, get_cached_model, get_torch_device, require_pkg, tensor_to_pils
import numpy as np
import torch
import cv2

CATEGORY = "llfnodes/图像处理/图像关键点与特征提取"


class DISKNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DISK",
            display_name="DISK",
            category=CATEGORY,
            inputs=[io.Image.Input("image"), io.Int.Input("max_keypoints", default=1024, min=16, max=8192)],
            outputs=[io.Image.Output("IMAGE"), io.String.Output("KEYPOINTS")],
        )

    @classmethod
    def execute(cls, image, max_keypoints):
        require_pkg("kornia", "pip install kornia")
        import kornia.feature as KF
        from torchvision.transforms.functional import to_tensor

        def loader(device):
            return KF.DISK.from_pretrained("depth").to(device).eval()

        model = get_cached_model("disk", loader)
        device = get_torch_device()
        outs, texts = [], []
        for pil in tensor_to_pils(image):
            img = to_tensor(pil.convert("RGB")).unsqueeze(0).to(device)
            with torch.inference_mode():
                feats = model(img, n=max_keypoints, pad_if_not_divisible=True)
            kpts = feats[0].keypoints.detach().cpu().numpy()
            canvas = np.asarray(pil.convert("RGB")).copy()
            for x, y in kpts:
                cv2.circle(canvas, (int(x), int(y)), 2, (255, 128, 0), -1)
            outs.append(array_to_image(canvas))
            texts.append("\n".join(f"{x:.1f},{y:.1f}" for x, y in kpts[:200]))
        return io.NodeOutput(torch.cat(outs, dim=0), "\n---\n".join(texts))
