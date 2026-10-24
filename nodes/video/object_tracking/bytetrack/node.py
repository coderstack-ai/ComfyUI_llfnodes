import numpy as np
from comfy_api.latest import io

from ....model_store import ensure_yolo_weight
from ...video_shared import draw_boxes_on_frames, frames_to_video, require_pkg, video_to_frames

CATEGORY = "llfnodes/视频处理/视频目标跟踪"


class ByteTrackNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.ByteTrack",
            display_name="ByteTrack",
            category=CATEGORY,
            description="YOLO + ByteTrack。检测权重位于 ComfyUI/models/detection/。",
            inputs=[
                io.Video.Input("video"),
                io.Combo.Input("det_model", options=["yolov8n.pt", "yolov8s.pt", "yolo11n.pt"], default="yolov8n.pt"),
                io.Float.Input("confidence", default=0.25, min=0.01, max=1.0, step=0.01),
            ],
            outputs=[io.Video.Output("VIDEO"), io.String.Output("TRACKS")],
        )

    @classmethod
    def execute(cls, video, det_model, confidence):
        require_pkg("ultralytics", "pip install ultralytics")
        from ultralytics import YOLO

        model = YOLO(str(ensure_yolo_weight(det_model)))
        frames = video_to_frames(video)
        tracks_vis, lines = [], []
        for i in range(frames.shape[0]):
            rgb = (frames[i].detach().cpu().numpy() * 255).astype(np.uint8)
            res = model.track(rgb[..., ::-1], persist=True, conf=confidence, tracker="bytetrack.yaml", verbose=False)[0]
            items = []
            if res.boxes is not None and len(res.boxes):
                for b in res.boxes:
                    xyxy = b.xyxy[0].tolist()
                    tid = int(b.id[0]) if b.id is not None else -1
                    cls_id = int(b.cls[0])
                    name = res.names.get(cls_id, str(cls_id))
                    items.append((*xyxy, f"id{tid}:{name}", (0, 255, 0)))
                    lines.append(f"f{i} id={tid} {name} {xyxy}")
            tracks_vis.append(items)
        return io.NodeOutput(frames_to_video(draw_boxes_on_frames(frames, tracks_vis), video), "\n".join(lines[:500]))
