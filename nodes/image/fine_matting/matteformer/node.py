import numpy as np
import torch
from comfy_api.latest import io

from ...cv_shared import tensor_to_pils

CATEGORY = "llfnodes/图像处理/精细透明度估计"


class MatteFormerNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.MatteFormer",
            display_name="MatteFormer",
            category=CATEGORY,
            description="MatteFormer 官方权重未内置；当前按 trimap 生成可运行 alpha（未知区取中值）。",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("trimap"),
                io.String.Input("model_path", default=""),
            ],
            outputs=[io.Mask.Output("ALPHA")],
        )

    @classmethod
    def execute(cls, image, trimap, model_path=""):
        # image unused except for batch size / spatial align
        t = trimap.unsqueeze(0) if trimap.ndim == 2 else trimap
        alphas = []
        pils = tensor_to_pils(image)
        for i, pil in enumerate(pils):
            m = t[min(i, t.shape[0] - 1)].detach().cpu().float().numpy()
            m = np.squeeze(m)
            if m.max() > 1.5:
                m = m / 255.0
            # classic trimap: 0=bg, 1=fg, (0,1)=unknown → 0.5
            alpha = np.where(m >= 0.9, 1.0, np.where(m <= 0.1, 0.0, 0.5)).astype(np.float32)
            if alpha.shape != (pil.height, pil.width):
                import cv2

                alpha = cv2.resize(alpha, (pil.width, pil.height), interpolation=cv2.INTER_LINEAR)
            alphas.append(torch.from_numpy(alpha)[None, ...])
        return io.NodeOutput(torch.cat(alphas, dim=0))
