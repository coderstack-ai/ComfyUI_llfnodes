import os

import folder_paths
from comfy_api.latest import InputImpl, io, ui


class VideoLoadNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return io.Schema(
            node_id="llfnodes.VideoLoadNode",
            display_name="视频加载",
            category="llfnodes/视频处理",
            has_intermediate_output=True,
            inputs=[
                io.Combo.Input("file", options=sorted(files), upload=io.UploadType.video),
            ],
            outputs=[io.Video.Output("VIDEO")],
        )

    @classmethod
    def execute(cls, file):
        video_path = folder_paths.get_annotated_filepath(file)
        source = InputImpl.VideoFromFile(video_path)
        name, _ = folder_paths.annotated_filepath(file)
        subfolder, _, filename = name.replace("\\", "/").rpartition("/")
        preview = ui.PreviewVideo([ui.SavedResult(filename, subfolder, io.FolderType.input)])
        return io.NodeOutput(source, ui=preview)

    @classmethod
    def fingerprint_inputs(cls, file):
        video_path = folder_paths.get_annotated_filepath(file)
        return os.path.getmtime(video_path)

    @classmethod
    def validate_inputs(cls, file):
        if not folder_paths.exists_annotated_filepath(file):
            return f"Invalid video file: {file}"
        return True
