"""HuggingFace Bucket helpers for reading/writing files."""

import json
import os
import tempfile
from typing import Any

from huggingface_hub import HfFileSystem
from loguru import logger

BUCKET_PREFIX = "buckets"


def _get_fs() -> HfFileSystem:
    return HfFileSystem()


def bucket_path(bucket: str, *parts: str) -> str:
    """Build a full HfFileSystem path for a bucket."""
    segments = [BUCKET_PREFIX, bucket, *parts]
    return "/".join(segments)


def read_json(bucket: str, path: str) -> dict[str, Any]:
    """Read a JSON file from a bucket."""
    fs = _get_fs()
    full = bucket_path(bucket, path)
    with fs.open(full, "r") as f:
        return json.load(f)


def write_json(bucket: str, path: str, data: dict[str, Any]) -> None:
    """Write a JSON file to a bucket (overwrites)."""
    fs = _get_fs()
    full = bucket_path(bucket, path)
    with fs.open(full, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {full}")


def list_audio_files(bucket: str, prefix: str = "") -> list[str]:
    """List all audio files in a bucket under a prefix.

    Returns relative paths within the bucket (e.g. 'batch-1/track.opus').
    """
    fs = _get_fs()
    base = bucket_path(bucket, prefix) if prefix else bucket_path(bucket)
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".opus"}
    results: list[str] = []

    for entry in fs.ls(base, detail=False):
        # entry is a full hffs path like 'buckets/user/repo/batch-1'
        if fs.isdir(entry):
            # recurse into subdirectories
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
    fs = _get_fs()
    dest = bucket_path(bucket, remote_path)
    fs.put(local_path, dest)
    logger.debug(f"Uploaded {local_path} -> {dest}")


def upload_directory(bucket: str, local_dir: str, remote_prefix: str) -> int:
    """Upload an entire local directory to the bucket. Returns file count."""
    fs = _get_fs()
    count = 0

    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            rel = os.path.relpath(local_path, local_dir)
            remote = f"{remote_prefix}/{rel}" if remote_prefix else rel
            dest = bucket_path(bucket, remote)
            fs.put(local_path, dest)
            count += 1

    logger.info(f"Uploaded {count} files to {bucket}/{remote_prefix}")
    return count


def file_exists(bucket: str, path: str) -> bool:
    """Check if a file exists in the bucket."""
    fs = _get_fs()
    return fs.exists(bucket_path(bucket, path))
