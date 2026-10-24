from comfy_api.latest import io
from ...cv_shared import get_cached_model, get_torch_device, tensor_to_pils
from ....model_store import hf_local

CATEGORY = "llfnodes/图像处理/图像描述与视觉理解"
MODELS = {
    "blip-base": "Salesforce/blip-image-captioning-base",
    "blip-large": "Salesforce/blip-image-captioning-large",
}


class BLIPNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.BLIP",
            display_name="BLIP",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS), default="blip-base"),
            ],
            outputs=[io.String.Output("CAPTION")],
        )

    @classmethod
    def execute(cls, image, model):
        from transformers import BlipForConditionalGeneration, BlipProcessor

        mid = MODELS[model]
        local = hf_local(mid)

        def loader(device):
            return (
                BlipProcessor.from_pretrained(local, local_files_only=True),
                BlipForConditionalGeneration.from_pretrained(local, local_files_only=True).to(device).eval(),
            )

        processor, net = get_cached_model(f"blip:{mid}", loader)
        device = get_torch_device()
        caps = []
        for pil in tensor_to_pils(image):
            inputs = processor(images=pil, return_tensors="pt").to(device)
            out = net.generate(**inputs, max_new_tokens=50)
            caps.append(processor.decode(out[0], skip_special_tokens=True))
        return io.NodeOutput("\n---\n".join(caps))
