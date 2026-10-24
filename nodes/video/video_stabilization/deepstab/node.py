import numpy as np
import torch
from comfy_api.latest import io
from ...video_shared import array_batch_to_image, frames_to_video, video_to_frames

CATEGORY = "llfnodes/视频处理/视频稳定"


class DeepStabNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.DeepStab",
            display_name="DeepStab",
            category=CATEGORY,
            description="DeepStab 风格视频稳定（OpenCV 仿射轨迹平滑近似）。",
            inputs=[
                io.Video.Input("video"),
                io.Float.Input("smoothing", default=30.0, min=1.0, max=200.0, step=1.0),
            ],
            outputs=[io.Video.Output("VIDEO")],
        )

    @classmethod
    def execute(cls, video, smoothing):
        import cv2

        frames = video_to_frames(video)
        imgs = [(frames[i].cpu().numpy() * 255).astype(np.uint8) for i in range(frames.shape[0])]
        if len(imgs) < 2:
            return io.NodeOutput(video)
        transforms = [np.zeros(3)]
        prev_gray = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2GRAY)
        for i in range(1, len(imgs)):
            gray = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
            if pts is None:
                transforms.append(np.zeros(3))
                prev_gray = gray
                continue
            pts2, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
            idx = status.reshape(-1) == 1
            mat = cv2.estimateAffinePartial2D(pts[idx], pts2[idx])[0]
            if mat is None:
                transforms.append(np.zeros(3))
            else:
                dx = mat[0, 2]
                dy = mat[1, 2]
                da = np.arctan2(mat[1, 0], mat[0, 0])
                transforms.append(np.array([dx, dy, da]))
            prev_gray = gray
        transforms = np.array(transforms)
        trajectory = np.cumsum(transforms, axis=0)
        smoothed = np.copy(trajectory)
        radius = int(max(1, smoothing))
        for i in range(3):
            smoothed[:, i] = np.convolve(trajectory[:, i], np.ones(radius) / radius, mode="same")
        difference = smoothed - trajectory
        outs = []
        h, w = imgs[0].shape[:2]
        for i, img in enumerate(imgs):
            dx, dy, da = difference[i]
            mat = np.array([[np.cos(da), -np.sin(da), dx], [np.sin(da), np.cos(da), dy]], dtype=np.float32)
            outs.append(cv2.warpAffine(img, mat, (w, h)))
        return io.NodeOutput(frames_to_video(array_batch_to_image(outs), video))
