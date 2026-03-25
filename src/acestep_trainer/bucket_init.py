"""Bucket initialization and template setup."""

from __future__ import annotations

import subprocess
import tempfile
from typing import Any

from loguru import logger

from acestep_trainer.bucket import _hf_url
from acestep_trainer.bucket import _run_hf
from acestep_trainer.bucket import write_json

TEMPLATE_DATASET: dict[str, Any] = {
    "metadata": {
        "name": "",
        "custom_tag": "",
        "tag_position": "prepend",
        "num_samples": 0,
        "all_instrumental": True,
        "genre_ratio": 50,
    },
    "samples": [],
}

TEMPLATE_README = """# {name}

ACE-Step 1.5 training dataset managed by [acestep-trainer](https://github.com/pedroapfilho/acestep-trainer).

## Structure

```
{bucket_id}/
├── batch-1/          # Audio files (.wav, .mp3, .flac, .ogg, .opus)
│   ├── track_001.opus
│   └── ...
├── batch-2/
│   └── ...
├── dataset.json      # Auto-generated state file (do not edit manually)
├── tensors/          # Auto-generated preprocessed tensors
│   ├── manifest.json
│   └── *.pt
└── labels_shard_*.json  # Temporary shard files during parallel labeling
```

## Audio requirements

- **Formats**: .wav, .mp3, .flac, .ogg, .opus
- **Sample rate**: any (resampled to 48kHz during preprocessing)
- **Duration**: up to 240 seconds per track
- **Organization**: group files in batch folders (batch-1, batch-2, etc.)

## Pipeline

1. **Scan**: `uv run acestep-train scan {bucket_id}`
2. **Label**: `uv run python scripts/submit_job.py label --bucket {bucket_id} --parallel 8`
3. **Merge**: `uv run acestep-train merge {bucket_id} --num-shards 8`
4. **Preprocess**: `uv run python scripts/submit_job.py preprocess --bucket {bucket_id}`
5. **Train**: `uv run python scripts/submit_job.py train --bucket {bucket_id} --output-repo <repo>`
"""


def bucket_exists(bucket_id: str) -> bool:
    """Check if a bucket exists."""
    result = subprocess.run(
        ["hf", "buckets", "list", bucket_id],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def create_bucket(bucket_id: str, private: bool = True) -> bool:
    """Create a new HF bucket."""
    args = ["hf", "buckets", "create", bucket_id]
    if private:
        args.append("--private")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to create bucket: {result.stderr}")
        return False
    logger.info(f"Created bucket: {bucket_id}")
    return True


def init_bucket(
    bucket_id: str,
    name: str,
    custom_tag: str = "",
    all_instrumental: bool = True,
    genre_ratio: int = 50,
) -> bool:
    """Initialize a bucket with template dataset.json and README."""
    dataset = dict(TEMPLATE_DATASET)
    dataset["metadata"] = {
        **TEMPLATE_DATASET["metadata"],
        "name": name,
        "custom_tag": custom_tag,
        "all_instrumental": all_instrumental,
        "genre_ratio": genre_ratio,
    }

    write_json(bucket_id, "dataset.json", dataset)

    readme_content = TEMPLATE_README.format(name=name, bucket_id=bucket_id)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(readme_content)
        tmp_path = f.name

    import os

    try:
        _run_hf("buckets", "cp", tmp_path, _hf_url(bucket_id, "README.md"))
        logger.info(f"Initialized bucket {bucket_id} with dataset.json and README.md")
        return True
    finally:
        os.unlink(tmp_path)
