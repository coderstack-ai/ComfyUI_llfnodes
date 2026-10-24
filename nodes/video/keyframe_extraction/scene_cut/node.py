import numpy as np
import torch
from comfy_api.latest import io
from ...video_shared import array_batch_to_image, video_to_frames

CATEGORY = "llfnodes/视频处理/视频关键帧提取"


class SceneCutNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.SceneCut",
            display_name="场景切换检测",
            category=CATEGORY,
            description="基于帧差直方图的场景切换/关键帧提取。",
            inputs=[
                io.Video.Input("video"),
                io.Float.Input("threshold", default=0.35, min=0.01, max=1.0, step=0.01),
                io.Int.Input("min_gap", default=8, min=1, max=240),
            ],
            outputs=[io.Image.Output("KEYFRAMES"), io.String.Output("INDICES")],
        )

    @classmethod
    def execute(cls, video, threshold, min_gap):
        frames = video_to_frames(video)
        idxs = [0]
        prev = None
        for i in range(frames.shape[0]):
            hist = np.histogram(
                (frames[i].mean(-1).cpu().numpy() * 255).astype(np.uint8),
                bins=64,
                range=(0, 255),
                density=True,
            )[0]
            if prev is not None:
                diff = 0.5 * np.abs(hist - prev).sum()
                if diff >= threshold and (i - idxs[-1]) >= min_gap:
                    idxs.append(i)
            prev = hist
        if idxs[-1] != frames.shape[0] - 1:
            idxs.append(frames.shape[0] - 1)
        key = frames[idxs]
        return io.NodeOutput(key, ",".join(str(i) for i in idxs))
