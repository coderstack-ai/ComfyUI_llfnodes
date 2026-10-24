"""Resolve / download models under ComfyUI's models directory (folder_paths.models_dir).

Never use a hardcoded drive path. Missing or corrupt weights are re-downloaded.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
import urllib.request
from pathlib import Path

import folder_paths

logger = logging.getLogger("llfnodes.model_store")

# folder_name -> relative dir under models_dir (created + registered if needed)
FOLDER_SPECS: dict[str, str] = {
    "optical_flow": "optical_flow",
    "detection": "detection",
    "action_recognition": "action_recognition",
    "video_mae": "video_mae",
    "upscale_models": "upscale_models",
    "background_removal": "background_removal",
    "frame_interpolation": "frame_interpolation",
    "diffusers": "diffusers",
    "clip_vision": "clip_vision",
    "llfnodes_hf": "llfnodes_hf",
}


def models_root() -> Path:
    return Path(folder_paths.models_dir)


def ensure_folder(folder_name: str, rel: str | None = None) -> Path:
    """Ensure ``<models_dir>/<rel>`` exists and is registered in folder_paths."""
    rel = rel or FOLDER_SPECS.get(folder_name, folder_name)
    path = models_root() / rel
    path.mkdir(parents=True, exist_ok=True)
    if folder_name not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(folder_name, str(path), is_default=True)
        # allow common weight extensions
        folder_paths.folder_names_and_paths[folder_name] = (
            folder_paths.folder_names_and_paths[folder_name][0],
            set(folder_paths.supported_pt_extensions) | {""},
        )
    else:
        # make sure our path is known
        folder_paths.add_model_folder_path(folder_name, str(path), is_default=False)
    return path


def register_llfnodes_folders() -> None:
    for name, rel in FOLDER_SPECS.items():
        ensure_folder(name, rel)


def _cli_tqdm(total: int | None, desc: str):
    """Command-line progress bar (stderr), visible in the ComfyUI console."""
    try:
        from tqdm import tqdm

        return tqdm(
            total=total if total and total > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
            leave=True,
            dynamic_ncols=True,
            file=None,  # stderr
        )
    except Exception:
        return None


def _enable_hf_cli_progress() -> None:
    try:
        from huggingface_hub.utils import enable_progress_bars

        enable_progress_bars()
    except Exception:
        pass


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    print(f"[llfnodes] Downloading: {url}", flush=True)
    print(f"[llfnodes]          -> {dest}", flush=True)
    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            pbar = _cli_tqdm(total if total > 0 else None, desc=dest.name)
            chunk_size = 1024 * 1024
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    if pbar is not None:
                        pbar.update(len(chunk))
            if pbar is not None:
                pbar.close()
        tmp.replace(dest)
        print(f"[llfnodes] Download complete: {dest.name}", flush=True)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def validate_torch_file(path: Path, min_bytes: int = 1024) -> bool:
    if not path.is_file() or path.stat().st_size < min_bytes:
        return False
    try:
        import torch

        torch.load(path, map_location="cpu", weights_only=True)
        return True
    except Exception:
        try:
            import torch

            torch.load(path, map_location="cpu", weights_only=False)
            return True
        except Exception as e:
            logger.warning("Corrupt weight %s: %s", path, e)
            return False


def _is_weight_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".bin", ".safetensors", ".pt", ".pth", ".ckpt"}


def validate_hf_dir(path: Path) -> bool:
    """Return True only when a local HF/diffusers checkout looks complete enough to load."""
    if not path.is_dir():
        return False

    model_index = path / "model_index.json"
    if model_index.exists():
        try:
            idx = json.loads(model_index.read_text(encoding="utf-8"))
        except Exception:
            return False
        for key, val in idx.items():
            if key.startswith("_"):
                continue
            # component entries are ["library", "ClassName"]; null means unused
            if not isinstance(val, (list, tuple)) or len(val) < 2 or val[0] is None:
                continue
            sub = path / key
            if not sub.is_dir():
                return False
            has_cfg = any(sub.glob("*config*.json")) or (sub / "config.json").exists()
            if not has_cfg:
                return False
            cls = str(val[1])
            # tokenizers / schedulers / processors don't need weight tensors
            if any(s in cls for s in ("Tokenizer", "Scheduler", "FeatureExtractor", "Processor", "ImageProcessor")):
                continue
            if not any(_is_weight_file(p) for p in sub.iterdir()):
                return False
        return True

    # transformers / preprocessor layout
    has_config = (path / "config.json").exists() or (path / "preprocessor_config.json").exists()
    if not has_config:
        return False
    return any(
        _is_weight_file(p)
        for p in path.rglob("*")
        if ".cache" not in p.parts
    )


def _is_transient_hf_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    keys = (
        "ssleoferror",
        "unexpected_eof",
        "ssl",
        "connection reset",
        "connection aborted",
        "timed out",
        "timeout",
        "max retries",
        "temporarily unavailable",
        "503",
        "502",
        "429",
    )
    return any(k in text for k in keys)


def ensure_file(
    folder_name: str,
    filename: str,
    *,
    url: str | None = None,
    validator=None,
    min_bytes: int = 1024,
) -> Path:
    """Return path to ``filename`` under the folder; download if missing/corrupt."""
    folder = ensure_folder(folder_name)
    dest = folder / filename
    check = validator or (lambda p: validate_torch_file(p, min_bytes=min_bytes))

    if dest.exists() and check(dest):
        return dest

    if dest.exists():
        logger.warning("Removing corrupt model: %s", dest)
        dest.unlink(missing_ok=True)

    # also check folder_paths lookup (other roots)
    existing = folder_paths.get_full_path(folder_name, filename)
    if existing:
        p = Path(existing)
        if check(p):
            return p
        logger.warning("Removing corrupt model: %s", p)
        p.unlink(missing_ok=True)

    if not url:
        raise RuntimeError(
            f"缺少模型 {folder_name}/{filename}（目录: {folder}），且未配置下载 URL。"
        )

    _download_url(url, dest)
    if not check(dest):
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"下载后校验失败: {dest}")
    return dest


def ensure_torchvision_weights(folder_name: str, filename: str, weights_enum) -> Path:
    """Download torchvision Weights enum url into ComfyUI models folder."""
    meta_path = ensure_folder(folder_name) / f"{Path(filename).stem}.meta.json"

    def _ok(p: Path) -> bool:
        return validate_torch_file(p)

    path = ensure_file(folder_name, filename, url=weights_enum.url, validator=_ok)
    if not meta_path.exists():
        meta = {
            "url": weights_enum.url,
            "categories": list(weights_enum.meta.get("categories", [])),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_categories_meta(folder_name: str, filename: str, fallback=None) -> list:
    meta_path = ensure_folder(folder_name) / f"{Path(filename).stem}.meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8")).get("categories", fallback or [])
        except Exception:
            pass
    return list(fallback or [])


def hf_local(repo_id: str, folder_name: str = "llfnodes_hf") -> str:
    """Ensure ``repo_id`` lives under ``models/<folder>/<owner--name>/`` and return that path.

    Call this before ``from_pretrained`` / ``pipeline`` so weights are not left only in
    ``~/.cache/huggingface``. Already-complete local dirs are reused (no re-download).
    """
    dirname = repo_id.replace("/", "--")
    return str(ensure_hf_repo(folder_name, dirname, repo_id))


def ensure_hf_repo(folder_name: str, dirname: str, repo_id: str, *, attempts: int = 6) -> Path:
    """Snapshot-download a HuggingFace repo into ``models/<folder>/<dirname>/``.

    Keeps partial downloads and retries on flaky SSL/network errors (serial workers).
    Respects ``HF_ENDPOINT`` (e.g. https://hf-mirror.com) if set in the environment.
    """
    folder = ensure_folder(folder_name)
    dest = folder / dirname
    if validate_hf_dir(dest):
        print(f"[llfnodes] Local model OK (skip download): {dest}", flush=True)
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        print(f"[llfnodes] Local model incomplete, resuming: {dest}", flush=True)
    print(f"[llfnodes] HF snapshot: {repo_id} -> {dest}", flush=True)
    _enable_hf_cli_progress()

    from huggingface_hub import snapshot_download

    last_err: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            # max_workers=1: parallel HF downloads often trip SSL EOF on unstable links
            # Prefer safetensors; keep configs/tokenizers. Skip ONNX/OpenVINO duplicates.
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest),
                local_dir_use_symlinks=False,
                max_workers=1,
                ignore_patterns=[
                    "*.onnx",
                    "*.pb",
                    "*.h5",
                    "*.msgpack",
                    "*.openvino*",
                    "flax_model*",
                    "tf_model*",
                ],
            )
            if validate_hf_dir(dest):
                print(f"[llfnodes] HF snapshot complete: {dirname}", flush=True)
                return dest
            last_err = RuntimeError(f"下载未完成或校验失败: {dest}")
            print(f"[llfnodes] HF incomplete after attempt {attempt}/{attempts}, retrying...", flush=True)
        except Exception as e:
            last_err = e
            if attempt >= attempts or not _is_transient_hf_error(e):
                break
            wait = min(2**attempt, 30)
            print(
                f"[llfnodes] HF download error (attempt {attempt}/{attempts}): {e}\n"
                f"[llfnodes] retry in {wait}s (partial files kept for resume)...",
                flush=True,
            )
            time.sleep(wait)

    err_text = str(last_err)
    if "GatedRepoError" in type(last_err).__name__ or "gated" in err_text.lower() or "403" in err_text:
        hint = (
            f"该模型为 gated 仓库（需在网页同意协议并登录）：https://huggingface.co/{repo_id}\n"
            "同意后执行：huggingface-cli login（或设置 HF_TOKEN），并尽量直连 huggingface.co（不要用无同步不全的镜像）。"
        )
    elif "404" in err_text or "Repository Not Found" in err_text:
        hint = f"仓库不存在或镜像未同步：{repo_id}。请核对 repo_id，或改用官方 huggingface.co（取消 HF_ENDPOINT）。"
    else:
        hint = (
            "网络/SSL 不稳定时可将环境变量 HF_ENDPOINT 设为镜像，例如 https://hf-mirror.com ，"
            "然后重启 ComfyUI 再试。"
        )
    raise RuntimeError(f"下载 HuggingFace 模型失败 {repo_id}: {last_err}\n{hint}") from last_err


def ensure_yolo_weight(filename: str = "yolov8n.pt") -> Path:
    """Place ultralytics YOLO weights under models/detection/."""
    folder = ensure_folder("detection")
    dest = folder / filename
    if dest.exists() and dest.stat().st_size > 1024:
        return dest

    # Prefer direct URL download so CLI tqdm progress is shown
    name = Path(filename).name
    url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}"
    try:
        _download_url(url, dest)
    except Exception as e:
        logger.warning("Direct YOLO download failed (%s), falling back to ultralytics: %s", url, e)
        from ultralytics import YOLO
        from ultralytics.utils.downloads import attempt_download_asset

        try:
            src = Path(attempt_download_asset(filename))
        except Exception:
            YOLO(filename)
            src = Path(filename)
            if not src.exists():
                raise RuntimeError(f"无法下载 YOLO 权重: {filename}") from e
        if dest.exists():
            dest.unlink()
        shutil.copy2(src, dest)

    if dest.stat().st_size < 1024:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"YOLO 权重异常: {dest}")
    return dest


def ensure_hf_file(repo_id: str, filename: str, dest: Path) -> Path:
    """Download a single HF file into ``dest`` with CLI tqdm progress."""
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_url

        url = hf_hub_url(repo_id=repo_id, filename=filename)
        _download_url(url, dest)
    except Exception as e:
        from huggingface_hub import hf_hub_download

        logger.warning("Direct HF file download failed, using hub cache: %s", e)
        _enable_hf_cli_progress()
        print(f"[llfnodes] HF download: {repo_id}/{filename}", flush=True)
        cached = hf_hub_download(repo_id=repo_id, filename=filename)
        if dest.exists():
            dest.unlink()
        shutil.copy2(cached, dest)
        print(f"[llfnodes] HF download complete: {dest.name}", flush=True)
    if dest.stat().st_size < 1024:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"HF 文件下载异常: {repo_id}/{filename}")
    return dest
