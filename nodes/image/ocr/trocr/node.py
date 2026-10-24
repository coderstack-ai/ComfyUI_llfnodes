from comfy_api.latest import io
from ...cv_shared import get_cached_model, get_torch_device, tensor_to_pils
from ....model_store import hf_local

CATEGORY = "llfnodes/图像处理/图像文字识别"
MODELS = {
    "base-printed": "microsoft/trocr-base-printed",
    "base-handwritten": "microsoft/trocr-base-handwritten",
    "large-printed": "microsoft/trocr-large-printed",
}


class TrOCRNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.TrOCR",
            display_name="TrOCR",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS), default="base-printed"),
            ],
            outputs=[io.String.Output("TEXT")],
        )

    @classmethod
    def execute(cls, image, model):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        mid = MODELS[model]
        local = hf_local(mid)

        def loader(device):
            return (
                TrOCRProcessor.from_pretrained(local, local_files_only=True),
                VisionEncoderDecoderModel.from_pretrained(local, local_files_only=True).to(device).eval(),
            )

        processor, net = get_cached_model(f"trocr:{mid}", loader)
        device = get_torch_device()
        texts = []
        for pil in tensor_to_pils(image):
            pixel_values = processor(images=pil.convert("RGB"), return_tensors="pt").pixel_values.to(device)
            ids = net.generate(pixel_values)
            texts.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
        return io.NodeOutput("\n---\n".join(texts))
