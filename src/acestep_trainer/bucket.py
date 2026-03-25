"""HuggingFace Bucket helpers using the standalone `hf` CLI.

All operations go through the `hf buckets` CLI (Rust binary) to avoid
Python huggingface_hub version conflicts with transformers.
"""

import json
import os
import re
import subprocess
import tempfile
from typing import Any

from loguru import logger


def _hf_url(bucket: str, path: str = "") -> str:
    """Build an hf:// URL for the bucket CLI."""
    if path:
        return f"hf://buckets/{bucket}/{path}"
    return f"hf://buckets/{bucket}"


def _run_hf(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run an hf CLI command."""
    cmd = ["hf", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"hf {' '.join(args)} failed: {result.stderr}")
    return result


def read_json(bucket: str, path: str) -> dict[str, Any]:
    """Read a JSON file from a bucket."""
    result = _run_hf("buckets", "cp", _hf_url(bucket, path), "-", check=False)
    if result.returncode != 0:
        raise FileNotFoundError(f"Cannot read {path} from bucket {bucket}: {result.stderr}")
    return json.loads(result.stdout)


def write_json(bucket: str, path: str, data: dict[str, Any]) -> None:
    """Write a JSON file to a bucket (overwrites)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path = f.name

    try:
        _run_hf("buckets", "cp", tmp_path, _hf_url(bucket, path))
        logger.info(f"Wrote {path} to bucket {bucket}")
    finally:
        os.unlink(tmp_path)


def list_audio_files(bucket: str, prefix: str = "") -> list[str]:
    """List all audio files in a bucket recursively.

    Uses `hf buckets list -R` and filters by audio extensions.
    Returns relative paths (e.g. 'batch-1/track.opus').
    """
    target = f"{bucket}/{prefix}" if prefix else bucket
    result = _run_hf("buckets", "list", target, "-R")

    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".opus"}
    files: list[str] = []

    for line in result.stdout.strip().splitlines():
        # hf buckets list output: "  2026-03-24 17:04:29  batch-1/file.opus"
        # or with -h flag: "  1.0MB  2026-03-24  batch-1/file.opus"
        line = line.strip()
        if not line:
            continue

        # Extract the path (last whitespace-separated token)
        parts = line.split()
        if not parts:
            continue
        path = parts[-1]

        # Skip directories (end with /)
        if path.endswith("/"):
            continue

        ext = os.path.splitext(path)[1].lower()
        if ext in audio_exts:
            files.append(path)

    return sorted(files)


def download_files(
    bucket: str,
    files: list[str],
    dest_dir: str,
) -> list[str]:
    """Download a list of bucket-relative paths to a local directory.

    Preserves directory structure. Returns list of local paths.
    """
    local_paths: list[str] = []

    for rel_path in files:
        local_path = os.path.join(dest_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        _run_hf("buckets", "cp", _hf_url(bucket, rel_path), local_path)
        local_paths.append(local_path)

    return local_paths


def upload_file(bucket: str, local_path: str, remote_path: str) -> None:
    """Upload a local file to the bucket."""
    _run_hf("buckets", "cp", local_path, _hf_url(bucket, remote_path))
    logger.debug(f"Uploaded {local_path} -> {remote_path}")


def upload_directory(bucket: str, local_dir: str, remote_prefix: str) -> int:
    """Upload an entire local directory to the bucket. Returns file count."""
    _run_hf("buckets", "sync", local_dir, _hf_url(bucket, remote_prefix))
    count = sum(1 for _, _, files in os.walk(local_dir) for _ in files)
    logger.info(f"Synced {count} files to {bucket}/{remote_prefix}")
    return count


def file_exists(bucket: str, path: str) -> bool:
    """Check if a file exists in the bucket by trying to read it."""
    result = _run_hf("buckets", "cp", _hf_url(bucket, path), "-", check=False)
    return result.returncode == 0
