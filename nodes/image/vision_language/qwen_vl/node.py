from comfy_api.latest import io
from ...cv_shared import get_cached_model, get_torch_device, tensor_to_pils
from ....model_store import hf_local

CATEGORY = "llfnodes/图像处理/图像描述与视觉理解"


class QwenVLNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.QwenVL",
            display_name="Qwen-VL",
            category=CATEGORY,
            description="Qwen2-VL 图像理解。",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("prompt", default="Describe this image.", multiline=True),
                io.Combo.Input(
                    "model",
                    options=["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"],
                    default="Qwen/Qwen2-VL-2B-Instruct",
                ),
            ],
            outputs=[io.String.Output("TEXT")],
        )

    @classmethod
    def execute(cls, image, prompt, model):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        local = hf_local(model)

        def loader(device):
            processor = AutoProcessor.from_pretrained(local, trust_remote_code=True, local_files_only=True)
            net = AutoModelForVision2Seq.from_pretrained(local, trust_remote_code=True, local_files_only=True).to(device).eval()
            return processor, net

        processor, net = get_cached_model(f"qwen-vl:{model}", loader)
        device = get_torch_device()
        texts = []
        for pil in tensor_to_pils(image):
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[text], images=[pil], return_tensors="pt").to(device)
            out = net.generate(**inputs, max_new_tokens=256)
            texts.append(processor.batch_decode(out, skip_special_tokens=True)[0])
        return io.NodeOutput("\n---\n".join(texts))
