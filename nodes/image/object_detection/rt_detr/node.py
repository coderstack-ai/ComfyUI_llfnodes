from comfy_api.latest import io
from ...cv_shared import draw_boxes, get_cached_model, get_torch_device, pils_to_tensor, tensor_to_pils
from ....model_store import hf_local

CATEGORY = "llfnodes/图像处理/目标检测"
MODELS = {"Rtdetr-l": "PekingU/rtdetr_r50vd_coco_o365", "Rtdetr-r50": "PekingU/rtdetr_r50vd"}


class RTDETRNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.RTDETR",
            display_name="RT-DETR",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS), default="Rtdetr-l"),
                io.Float.Input("threshold", default=0.5, min=0.01, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output("IMAGE"), io.String.Output("DETECTIONS")],
        )

    @classmethod
    def execute(cls, image, model, threshold):
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        mid = MODELS[model]
        local = hf_local(mid)

        def loader(device):
            return (
                RTDetrImageProcessor.from_pretrained(local, local_files_only=True),
                RTDetrForObjectDetection.from_pretrained(local, local_files_only=True).to(device).eval(),
            )

        processor, net = get_cached_model(f"rtdetr:{mid}", loader)
        device = get_torch_device()
        outs, texts = [], []
        for pil in tensor_to_pils(image):
            inputs = {k: v.to(device) for k, v in processor(images=pil, return_tensors="pt").items()}
            outputs = net(**inputs)
            r = processor.post_process_object_detection(
                outputs, target_sizes=[pil.size[::-1]], threshold=threshold
            )[0]
            boxes = r["boxes"].detach().cpu().tolist()
            scores = r["scores"].detach().cpu().tolist()
            labels = [net.config.id2label.get(int(i), str(int(i))) for i in r["labels"].tolist()]
            outs.append(draw_boxes(pil, boxes, labels, scores))
            texts.append("\n".join(f"{l}:{s:.3f}" for l, s in zip(labels, scores)))
        return io.NodeOutput(pils_to_tensor(outs), "\n---\n".join(texts))
