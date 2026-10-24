import cv2
import numpy as np
import torch
from comfy_api.latest import io

from ...cv_shared import array_to_image, get_cached_model, get_torch_device, require_pkg, tensor_to_pils

CATEGORY = "llfnodes/图像处理/图像关键点与特征提取"


def _opencv_sift_keypoints(pil, max_keypoints: int):
    gray = np.asarray(pil.convert("L"))
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    kps = sift.detect(gray, None)
    kps = sorted(kps, key=lambda k: k.response, reverse=True)[:max_keypoints]
    return np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)


def _load_superpoint(device, max_keypoints: int):
    """kornia<=0.7 had KF.SuperPoint; 0.8+ removed it — try leftovers then KeyNet."""
    import kornia.feature as KF

    if hasattr(KF, "SuperPoint"):
        return ("superpoint", KF.SuperPoint(num_filters=64, max_num_keypoints=max_keypoints).to(device).eval())

    # KeyNet is the closest built-in learned detector still shipped in kornia 0.8
    detector = KF.KeyNetDetector(pretrained=True, num_features=max_keypoints).to(device).eval()
    return ("keynet", detector)


class SuperPointNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.SuperPoint",
            display_name="SuperPoint",
            category=CATEGORY,
            description="关键点检测。kornia 0.8+ 无 SuperPoint，自动改用 KeyNet；再失败则 OpenCV SIFT。",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("max_keypoints", default=1024, min=16, max=8192),
            ],
            outputs=[io.Image.Output("IMAGE"), io.String.Output("KEYPOINTS")],
        )

    @classmethod
    def execute(cls, image, max_keypoints):
        require_pkg("kornia", "pip install kornia")
        from torchvision.transforms.functional import to_tensor

        device = get_torch_device()
        backend = "opencv_sift"
        model = None
        try:

            def loader(dev):
                return _load_superpoint(dev, max_keypoints)

            backend, model = get_cached_model(f"superpoint-like:{max_keypoints}", loader)
        except Exception:
            backend, model = "opencv_sift", None

        outs, texts = [], []
        for pil in tensor_to_pils(image):
            if backend == "superpoint" and model is not None:
                img = to_tensor(pil.convert("L")).unsqueeze(0).to(device)
                with torch.inference_mode():
                    out = model({"image": img})
                kpts = out["keypoints"][0].detach().cpu().numpy()
            elif backend == "keynet" and model is not None:
                import kornia.feature as KF

                img = to_tensor(pil.convert("L")).unsqueeze(0).to(device)
                with torch.inference_mode():
                    lafs, _ = model(img)
                kpts = KF.get_laf_center(lafs)[0].detach().cpu().numpy()
            else:
                kpts = _opencv_sift_keypoints(pil, max_keypoints)

            kpts = kpts[:max_keypoints]
            canvas = np.asarray(pil.convert("RGB")).copy()
            for x, y in kpts:
                cv2.circle(canvas, (int(x), int(y)), 2, (0, 255, 0), -1)
            outs.append(array_to_image(canvas))
            note = f"# backend={backend}\n"
            texts.append(note + "\n".join(f"{x:.1f},{y:.1f}" for x, y in kpts[:200]))
        return io.NodeOutput(torch.cat(outs, dim=0), "\n---\n".join(texts))
