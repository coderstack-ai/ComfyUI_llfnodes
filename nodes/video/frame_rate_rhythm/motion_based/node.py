import numpy as np
import torch
from comfy_api.latest import io
from ...video_shared import video_meta, video_to_frames

CATEGORY = "llfnodes/视频处理/视频帧率与节奏分析"


class MotionRhythmNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.MotionRhythm",
            display_name="运动估计节奏分析",
            category=CATEGORY,
            description="基于相邻帧运动能量的帧率/节奏分析。",
            inputs=[
                io.Video.Input("video"),
            ],
            outputs=[io.String.Output("ANALYSIS"), io.Image.Output("MOTION_CURVE")],
        )

    @classmethod
    def execute(cls, video):
        import cv2

        frames = video_to_frames(video)
        fps, _, _ = video_meta(video)
        fps_f = float(fps)
        energies = []
        prev = None
        for i in range(frames.shape[0]):
            g = (frames[i].mean(-1).cpu().numpy() * 255).astype(np.uint8)
            if prev is not None:
                energies.append(float(np.mean(np.abs(g.astype(np.float32) - prev.astype(np.float32))) / 255.0))
            prev = g
        energies = np.array(energies, dtype=np.float32) if energies else np.array([0.0], dtype=np.float32)
        mean_e = float(energies.mean())
        peak = int(energies.argmax()) if len(energies) else 0
        # Draw simple curve image
        w, h = 512, 128
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if len(energies) > 1:
            xs = np.linspace(0, w - 1, len(energies)).astype(int)
            ys = (h - 1 - (energies / (energies.max() + 1e-6) * (h - 10))).astype(int)
            for i in range(1, len(xs)):
                cv2.line(canvas, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), (0, 255, 128), 2)
        info = (
            f"fps={fps_f:.3f}\nframes={frames.shape[0]}\n"
            f"mean_motion={mean_e:.4f}\npeak_frame={peak + 1}\n"
            f"suggested_keep_ratio={min(1.0, max(0.1, mean_e * 2)):.3f}"
        )
        curve = torch.from_numpy(canvas.astype(np.float32) / 255.0)[None, ...]
        return io.NodeOutput(info, curve)
