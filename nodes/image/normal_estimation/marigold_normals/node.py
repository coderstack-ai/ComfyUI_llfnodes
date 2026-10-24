from comfy_api.latest import io

from ..common import CATEGORY, run_marigold_normals

MODELS = {
    "v1.1": "prs-eth/marigold-normals-v1-1",
    "v0.1": "prs-eth/marigold-normals-v0-1",
}


class MarigoldNormalsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="llfnodes.MarigoldNormals",
            display_name="Marigold Normals",
            category=CATEGORY,
            description="Marigold 扩散表面法线估计（diffusers）。",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(MODELS.keys()), default="v1.1"),
                io.Int.Input("steps", default=4, min=1, max=50, step=1),
                io.Int.Input("ensemble_size", default=1, min=1, max=10, step=1),
            ],
            outputs=[io.Image.Output("NORMAL")],
        )

    @classmethod
    def execute(cls, image, model, steps, ensemble_size):
        out = run_marigold_normals(
            MODELS[model],
            image,
            steps=steps,
            ensemble_size=ensemble_size,
        )
        return io.NodeOutput(out)
