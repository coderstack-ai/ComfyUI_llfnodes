from comfy_api.latest import io
from ...cv_shared import draw_boxes, get_cached_model, get_torch_device, pils_to_tensor, tensor_to_pils
from ....model_store import hf_local

CATEGORY = "llfnodes/图像处理/目标检测"
MODELS = {"tiny": "IDEA-Research/grounding-dino-tiny", "base": "IDEA-Research/grounding-dino-base"}


class GroundingDINONode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.GroundingDINO",
            display_name="Grounding DINO",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.String.Input("prompt", default="person . car . dog", multiline=True),
                io.Combo.Input("model", options=list(MODELS), default="tiny"),
                io.Float.Input("box_threshold", default=0.35, min=0.01, max=1.0, step=0.01),
                io.Float.Input("text_threshold", default=0.25, min=0.01, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output("IMAGE"), io.String.Output("DETECTIONS")],
        )

    @classmethod
    def execute(cls, image, prompt, model, box_threshold, text_threshold):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        mid = MODELS[model]
        local = hf_local(mid)

        def loader(device):
            return (
                AutoProcessor.from_pretrained(local, local_files_only=True),
                AutoModelForZeroShotObjectDetection.from_pretrained(local, local_files_only=True).to(device).eval(),
            )

        processor, net = get_cached_model(f"gdino:{mid}", loader)
        device = get_torch_device()
        outs, texts = [], []
        for pil in tensor_to_pils(image):
            inputs = processor(images=pil, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = net(**inputs)
            # transformers>=4.4x renamed box_threshold → threshold
            r = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[pil.size[::-1]],
            )[0]
            boxes = r["boxes"].detach().cpu().tolist()
            scores = r["scores"].detach().cpu().tolist()
            labels = r.get("text_labels", r.get("labels", ["obj"] * len(boxes)))
            if hasattr(labels, "tolist"):
                labels = labels.tolist()
            outs.append(draw_boxes(pil, boxes, labels, scores))
            texts.append("\n".join(f"{l}:{s:.3f}" for l, s in zip(labels, scores)))
        return io.NodeOutput(pils_to_tensor(outs), "\n---\n".join(texts))
