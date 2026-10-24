from comfy_api.latest import io

from ....model_store import ensure_yolo_weight
from ...cv_shared import draw_boxes, get_cached_model, pils_to_tensor, require_pkg, tensor_to_pils

CATEGORY = "llfnodes/图像处理/目标检测"
MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov10n.pt", "yolo11n.pt"]


class YOLONode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.YOLO",
            display_name="YOLO",
            category=CATEGORY,
            description="YOLO 检测。权重位于 ComfyUI/models/detection/，缺失时自动下载。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=MODELS, default="yolov8n.pt"),
                io.Float.Input("confidence", default=0.25, min=0.01, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output("IMAGE"), io.String.Output("DETECTIONS")],
        )

    @classmethod
    def execute(cls, image, model, confidence):
        require_pkg("ultralytics", "pip install ultralytics")
        from ultralytics import YOLO

        weight = str(ensure_yolo_weight(model))
        yolo = get_cached_model(f"yolo:{weight}", lambda d: YOLO(weight))
        outs, texts = [], []
        for pil in tensor_to_pils(image):
            res = yolo.predict(pil, conf=confidence, verbose=False)[0]
            boxes, labels, scores, lines = [], [], [], []
            if res.boxes is not None and len(res.boxes):
                for b in res.boxes:
                    xyxy = b.xyxy[0].tolist()
                    cls_id = int(b.cls[0])
                    score = float(b.conf[0])
                    name = res.names.get(cls_id, str(cls_id))
                    boxes.append(xyxy)
                    labels.append(name)
                    scores.append(score)
                    lines.append(f"{name}:{score:.3f}")
            outs.append(draw_boxes(pil, boxes, labels, scores))
            texts.append("\n".join(lines))
        return io.NodeOutput(pils_to_tensor(outs), "\n---\n".join(texts))
