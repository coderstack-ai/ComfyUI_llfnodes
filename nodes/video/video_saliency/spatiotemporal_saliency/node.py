import numpy as np
import torch
from comfy_api.latest import io
from ...video_shared import array_batch_to_image, frames_to_video, video_to_frames

CATEGORY = "llfnodes/视频处理/视频显著性检测"


class SpatiotemporalSaliencyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.SpatiotemporalSaliency",
            display_name="时空显著性",
            category=CATEGORY,
            description="基于帧差 + 空间对比的轻量时空显著性。",
            inputs=[io.Video.Input("video")],
            outputs=[io.Video.Output("VIDEO"), io.Image.Output("SALIENCY")],
        )

    @classmethod
    def execute(cls, video):
        import cv2

        frames = video_to_frames(video)
        maps, overlays = [], []
        prev = None
        for i in range(frames.shape[0]):
            rgb = (frames[i].cpu().numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            spatial = cv2.GaussianBlur(gray, (0, 0), 8)
            spatial = np.abs(gray - spatial)
            temporal = np.abs(gray - prev) if prev is not None else np.zeros_like(gray)
            sal = spatial * 0.5 + temporal * 0.5
            sal = sal / (sal.max() + 1e-6)
            heat = cv2.applyColorMap((sal * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            blend = (rgb * 0.45 + heat * 0.55).astype(np.uint8)
            maps.append(np.repeat(sal[..., None], 3, axis=-1))
            overlays.append(blend)
            prev = gray
        return io.NodeOutput(
            frames_to_video(array_batch_to_image(overlays), video),
            array_batch_to_image(maps),
        )
