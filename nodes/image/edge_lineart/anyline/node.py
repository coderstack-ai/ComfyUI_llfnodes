import numpy as np
import torch
from comfy_api.latest import io

from ..common import (
    CATEGORY,
    array_to_image,
    ensure_controlnet_aux,
    get_cached_model,
    run_controlnet_detector,
)

LINEART_MODES = [
    "lineart_standard",
    "lineart_realistic",
    "lineart_anime",
    "manga_line",
]


def _intensity_mask(image_array: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    mask = image_array[:, :, 0]
    mask = np.where((mask >= lower_bound) & (mask <= upper_bound), mask, 0)
    return np.expand_dims(mask, 2).repeat(3, axis=2)


def _combine_layers(base_layer: np.ndarray, top_layer: np.ndarray) -> np.ndarray:
    mask = top_layer.astype(bool)
    temp = 1 - (1 - top_layer) * (1 - base_layer)
    return base_layer * (~mask) + temp * mask


class AnyLineNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.AnyLine",
            display_name="AnyLine",
            category=CATEGORY,
            description="AnyLine 线稿提取（MTEED + Lineart 融合）。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("merge_with_lineart", options=LINEART_MODES, default="lineart_standard"),
                io.Int.Input("resolution", default=1280, min=64, max=2048, step=8),
                io.Float.Input("lineart_lower_bound", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("lineart_upper_bound", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("object_min_size", default=36, min=1, max=1024, step=1),
                io.Int.Input("object_connectivity", default=1, min=1, max=8, step=1),
            ],
            outputs=[io.Image.Output("EDGE")],
        )

    @classmethod
    def execute(
        cls,
        image,
        merge_with_lineart,
        resolution,
        lineart_lower_bound,
        lineart_upper_bound,
        object_min_size,
        object_connectivity,
    ):
        ensure_controlnet_aux()
        from custom_controlnet_aux.teed import TEDDetector
        from skimage import morphology

        def loader_mteed(device):
            return TEDDetector.from_pretrained(
                "TheMistoAI/MistoLine", "MTEED.pth", subfolder="Anyline"
            ).to(device)

        mteed = get_cached_model("anyline-mteed", loader_mteed)
        mteed_batch = run_controlnet_detector(mteed, image, detect_resolution=resolution)

        if merge_with_lineart == "lineart_standard":
            from custom_controlnet_aux.lineart_standard import LineartStandardDetector

            lineart_detector = LineartStandardDetector()
            lineart_batch = run_controlnet_detector(
                lineart_detector,
                image,
                detect_resolution=resolution,
                guassian_sigma=2,
                intensity_threshold=3,
            )
        else:
            from custom_controlnet_aux.lineart import LineartDetector
            from custom_controlnet_aux.lineart_anime import LineartAnimeDetector
            from custom_controlnet_aux.manga_line import LineartMangaDetector

            # keep spelling compatible with controlnet_aux combo typo "realisitic"
            mode = "lineart_realisitic" if merge_with_lineart == "lineart_realistic" else merge_with_lineart
            detector_cls = {
                "lineart_realisitic": LineartDetector,
                "lineart_anime": LineartAnimeDetector,
                "manga_line": LineartMangaDetector,
            }[mode]

            def loader_lineart(device):
                return detector_cls.from_pretrained().to(device)

            lineart_detector = get_cached_model(f"anyline-lineart:{mode}", loader_lineart)
            lineart_batch = run_controlnet_detector(
                lineart_detector,
                image,
                detect_resolution=resolution,
            )

        outs = []
        for i in range(image.shape[0]):
            lineart_np = lineart_batch[i].detach().cpu().numpy()
            mteed_np = mteed_batch[i].detach().cpu().numpy()
            masked = _intensity_mask(lineart_np, lineart_lower_bound, lineart_upper_bound)
            cleaned = morphology.remove_small_objects(
                masked.astype(bool),
                min_size=object_min_size,
                connectivity=object_connectivity,
            )
            masked = masked * cleaned
            combined = _combine_layers(mteed_np, masked)
            outs.append(array_to_image(combined))

        return io.NodeOutput(torch.cat(outs, dim=0))
