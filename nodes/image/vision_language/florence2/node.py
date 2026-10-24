from comfy_api.latest import io
from ...cv_shared import get_cached_model, get_torch_device, tensor_to_pils
from ....model_store import hf_local

CATEGORY = "llfnodes/图像处理/图像描述与视觉理解"
MODELS = {
    "base": "microsoft/Florence-2-base",
    "large": "microsoft/Florence-2-large",
}


class Florence2Node(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.Florence2",
            display_name="Florence-2",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS), default="base"),
                io.String.Input("task", default="<CAPTION>"),
            ],
            outputs=[io.String.Output("TEXT")],
        )

    @classmethod
    def execute(cls, image, model, task):
        from transformers import AutoModelForCausalLM, AutoProcessor

        mid = MODELS[model]
        local = hf_local(mid)

        def loader(device):
            processor = AutoProcessor.from_pretrained(local, trust_remote_code=True, local_files_only=True)
            net = AutoModelForCausalLM.from_pretrained(local, trust_remote_code=True, local_files_only=True).to(device).eval()
            return processor, net

        processor, net = get_cached_model(f"florence2:{mid}", loader)
        device = get_torch_device()
        texts = []
        for pil in tensor_to_pils(image):
            inputs = processor(text=task, images=pil, return_tensors="pt").to(device)
            ids = net.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
            )
            texts.append(processor.batch_decode(ids, skip_special_tokens=False)[0])
        return io.NodeOutput("\n---\n".join(texts))
