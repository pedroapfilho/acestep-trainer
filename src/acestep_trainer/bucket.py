"""HuggingFace Bucket helpers for reading/writing files.

Uses `hf buckets cp` CLI for writes (most reliable for bucket storage)
and HfFileSystem for reads and directory listing.
"""

import json
import os
import subprocess
import tempfile
from typing import Any

from huggingface_hub import HfFileSystem
from loguru import logger

BUCKET_PREFIX = "buckets"


def _get_fs() -> HfFileSystem:
    return HfFileSystem()


def _hf_url(bucket: str, path: str) -> str:
    """Build an hf:// URL for the bucket CLI."""
    return f"hf://buckets/{bucket}/{path}"


def bucket_path(bucket: str, *parts: str) -> str:
    """Build a full HfFileSystem path for a bucket."""
    segments = [BUCKET_PREFIX, bucket, *parts]
    return "/".join(segments)


def read_json(bucket: str, path: str) -> dict[str, Any]:
    """Read a JSON file from a bucket."""
    result = subprocess.run(
        ["hf", "buckets", "cp", _hf_url(bucket, path), "-"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FileNotFoundError(f"Cannot read {path} from bucket {bucket}: {result.stderr}")
    return json.loads(result.stdout)


def write_json(bucket: str, path: str, data: dict[str, Any]) -> None:
    """Write a JSON file to a bucket (overwrites)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["hf", "buckets", "cp", tmp_path, _hf_url(bucket, path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to write {path}: {result.stderr}")
        logger.info(f"Wrote {path} to bucket {bucket}")
    finally:
        os.unlink(tmp_path)


def list_audio_files(bucket: str, prefix: str = "") -> list[str]:
    """List all audio files in a bucket under a prefix.

    Returns relative paths within the bucket (e.g. 'batch-1/track.opus').
    """
    fs = _get_fs()
    base = bucket_path(bucket, prefix) if prefix else bucket_path(bucket)
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".opus"}
    results: list[str] = []

    for entry in fs.ls(base, detail=False):
        if fs.isdir(entry):
            sub_prefix = entry.removeprefix(f"{BUCKET_PREFIX}/{bucket}/")
            results.extend(list_audio_files(bucket, sub_prefix))
        else:
            ext = os.path.splitext(entry)[1].lower()
            if ext in audio_exts:
                rel = entry.removeprefix(f"{BUCKET_PREFIX}/{bucket}/")
                results.append(rel)

    return sorted(results)


def download_files(
    bucket: str,
    files: list[str],
    dest_dir: str,
) -> list[str]:
    """Download a list of bucket-relative paths to a local directory.

    Preserves directory structure. Returns list of local paths.
    """
    fs = _get_fs()
    local_paths: list[str] = []

    for rel_path in files:
        local_path = os.path.join(dest_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        src = bucket_path(bucket, rel_path)
        fs.get(src, local_path)
        local_paths.append(local_path)

    return local_paths


def upload_file(bucket: str, local_path: str, remote_path: str) -> None:
    """Upload a local file to the bucket."""
    result = subprocess.run(
        ["hf", "buckets", "cp", local_path, _hf_url(bucket, remote_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to upload {local_path}: {result.stderr}")
    logger.debug(f"Uploaded {local_path} -> {remote_path}")


def upload_directory(bucket: str, local_dir: str, remote_prefix: str) -> int:
    """Upload an entire local directory to the bucket. Returns file count."""
    result = subprocess.run(
        ["hf", "buckets", "sync", local_dir, _hf_url(bucket, remote_prefix)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to sync {local_dir}: {result.stderr}")

    # Count files uploaded
    count = sum(1 for _, _, files in os.walk(local_dir) for _ in files)
    logger.info(f"Synced {count} files to {bucket}/{remote_prefix}")
    return count


def file_exists(bucket: str, path: str) -> bool:
    """Check if a file exists in the bucket by trying to read it."""
    result = subprocess.run(
        ["hf", "buckets", "cp", _hf_url(bucket, path), "-"],
        capture_output=True,
    )
    return result.returncode == 0
