import hashlib
import os

import folder_paths
import node_helpers
import numpy as np
import torch
from comfy_api.latest import io
from PIL import Image, ImageOps, ImageSequence


class ImageLoadNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return io.Schema(
            node_id="llfnodes.ImageLoadNode",
            display_name="图像加载",
            category="llfnodes/图像处理",
            inputs=[
                io.Combo.Input("image", options=sorted(files), upload=io.UploadType.image),
            ],
            outputs=[
                io.Image.Output("IMAGE"),
                io.Mask.Output("MASK"),
            ],
        )

    @classmethod
    def execute(cls, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w = h = None

        for frame in ImageSequence.Iterator(img):
            frame = node_helpers.pillow(ImageOps.exif_transpose, frame)
            rgb = frame.convert("RGB")

            if w is None:
                w, h = rgb.size
            if rgb.size != (w, h):
                continue

            image_t = torch.from_numpy(np.array(rgb).astype(np.float32) / 255.0)[None,]
            if "A" in frame.getbands():
                mask = 1.0 - torch.from_numpy(np.array(frame.getchannel("A")).astype(np.float32) / 255.0)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image_t)
            output_masks.append(mask.unsqueeze(0))

        return io.NodeOutput(torch.cat(output_images, dim=0), torch.cat(output_masks, dim=0))

    @classmethod
    def fingerprint_inputs(cls, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def validate_inputs(cls, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True
