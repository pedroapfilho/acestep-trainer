#!/usr/bin/env python3
"""Phase 1: Auto-label audio tracks from HF bucket using ACE-Step LLM.

Designed to run on HuggingFace Jobs with GPU. Downloads audio in batches,
runs the DiT audio encoder + LLM labeling, updates dataset.json in the bucket.

Supports sharded parallel execution: use --shard-id and --num-shards to split
work across multiple concurrent jobs. Each shard writes to a separate file
(labels_shard_{id}.json) to avoid write conflicts.

Usage (single job):
    python scripts/label.py --bucket pedroapfilho/lofi-tracks

Usage (parallel — 4 shards):
    python scripts/label.py --bucket ... --shard-id 0 --num-shards 4
    python scripts/label.py --bucket ... --shard-id 1 --num-shards 4
    python scripts/label.py --bucket ... --shard-id 2 --num-shards 4
    python scripts/label.py --bucket ... --shard-id 3 --num-shards 4

After all shards complete, run: uv run acestep-train merge <bucket>
"""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import Any

from loguru import logger

from acestep_trainer.bucket import download_files
from acestep_trainer.bucket import list_audio_files
from acestep_trainer.bucket import write_json
from acestep_trainer.handler import ensure_sys_path
from acestep_trainer.handler import init_dit_handler
from acestep_trainer.handler import init_llm_handler
from acestep_trainer.state import DatasetState
from acestep_trainer.state import SampleState
from acestep_trainer.state import load_state
from acestep_trainer.state import save_state
from acestep_trainer.state import sync_files_to_state


def label_batch(
    dit_handler: Any,
    llm_handler: Any,
    state: DatasetState,
    bucket: str,
    batch: list[SampleState],
    work_dir: str,
) -> int:
    """Download and label a batch of samples. Returns count of newly labeled."""
    from acestep.training.dataset_builder_modules.builder import (
        DatasetBuilder,  # type: ignore[import-untyped]
    )
    from acestep.training.dataset_builder_modules.models import (
        AudioSample,  # type: ignore[import-untyped]
    )

    files_to_download = [s.file for s in batch]
    local_paths = download_files(bucket, files_to_download, work_dir)

    builder = DatasetBuilder()
    for sample_state, local_path in zip(batch, local_paths):
        builder.samples.append(
            AudioSample(
                audio_path=local_path,
                filename=os.path.basename(local_path),
                is_instrumental=True,
            )
        )

    _, status = builder.label_all_samples(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        format_lyrics=False,
        transcribe_lyrics=False,
        skip_metas=False,
        only_unlabeled=False,
    )
    logger.info(f"Labeling result: {status}")

    labeled_count = 0
    for sample_state, builder_sample in zip(batch, builder.samples):
        if builder_sample.labeled:
            state.mark_labeled(
                sample_state.file,
                caption=builder_sample.caption,
                genre=builder_sample.genre,
                lyrics=builder_sample.lyrics,
                bpm=builder_sample.bpm,
                keyscale=builder_sample.keyscale,
                timesignature=builder_sample.timesignature,
                language=builder_sample.language,
                is_instrumental=builder_sample.is_instrumental,
            )
            labeled_count += 1

    return labeled_count


def save_shard(bucket: str, shard_id: int, state: DatasetState) -> None:
    """Save only the labeled samples from this shard to a shard file."""
    labeled = state.get_by_status("labeled")
    shard_data: dict[str, Any] = {
        "shard_id": shard_id,
        "samples": [s.to_dict() for s in labeled],
    }
    shard_path = f"labels_shard_{shard_id}.json"
    write_json(bucket, shard_path, shard_data)
    logger.info(f"Saved shard {shard_id}: {len(labeled)} labeled samples -> {shard_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label audio tracks from HF bucket")
    parser.add_argument("--bucket", required=True, help="HF bucket name (user/repo)")
    parser.add_argument("--batch-size", type=int, default=50, help="Files per batch")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to label (0=all)")
    parser.add_argument("--save-every", type=int, default=1, help="Save state every N batches")
    parser.add_argument(
        "--shard-id", type=int, default=-1, help="Shard ID for parallel execution (-1=single job)"
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of parallel shards")
    parsed = parser.parse_args()

    bucket: str = parsed.bucket
    batch_size: int = parsed.batch_size
    max_samples: int = parsed.max_samples
    save_every: int = parsed.save_every
    shard_id: int = parsed.shard_id
    num_shards: int = parsed.num_shards

    is_sharded = shard_id >= 0 and num_shards > 1

    logger.info(f"Starting labeling for bucket: {bucket}")
    if is_sharded:
        logger.info(f"Shard mode: {shard_id + 1}/{num_shards}")

    state = load_state(bucket)

    logger.info("Scanning bucket for audio files...")
    audio_files = list_audio_files(bucket)
    logger.info(f"Found {len(audio_files)} audio files in bucket")

    if not is_sharded or shard_id == 0:
        new_count = sync_files_to_state(bucket, state, audio_files)
        if new_count > 0:
            save_state(bucket, state)

    unlabeled = state.get_by_status("unlabeled")
    if not unlabeled:
        logger.info("All samples are already labeled!")
        return

    if is_sharded:
        unlabeled = [s for i, s in enumerate(unlabeled) if i % num_shards == shard_id]
        logger.info(f"Shard {shard_id}: {len(unlabeled)} samples assigned")

    if max_samples > 0:
        unlabeled = unlabeled[:max_samples]

    logger.info(f"Labeling {len(unlabeled)} samples in batches of {batch_size}")

    logger.info("Initializing ACE-Step models...")
    ensure_sys_path()
    dit_handler = init_dit_handler()
    llm_handler = init_llm_handler()

    if is_sharded:
        shard_state = DatasetState(
            name=state.name,
            custom_tag=state.custom_tag,
            tag_position=state.tag_position,
            genre_ratio=state.genre_ratio,
            all_instrumental=state.all_instrumental,
            samples=[SampleState(file=s.file) for s in unlabeled],
        )
    else:
        shard_state = state

    total_labeled = 0
    for batch_idx in range(0, len(unlabeled), batch_size):
        batch_samples = unlabeled[batch_idx : batch_idx + batch_size]
        batch_state = (
            shard_state.samples[batch_idx : batch_idx + batch_size] if is_sharded else batch_samples
        )
        batch_num = batch_idx // batch_size + 1
        total_batches = (len(unlabeled) + batch_size - 1) // batch_size

        logger.info(f"Batch {batch_num}/{total_batches}: {len(batch_state)} samples")

        with tempfile.TemporaryDirectory(prefix="acestep_label_") as work_dir:
            labeled = label_batch(
                dit_handler, llm_handler, shard_state, bucket, batch_state, work_dir
            )
            total_labeled += labeled
            logger.info(f"Batch {batch_num}: labeled {labeled}/{len(batch_state)}")

        if batch_num % save_every == 0:
            if is_sharded:
                save_shard(bucket, shard_id, shard_state)
            else:
                save_state(bucket, shard_state)
            logger.info(f"Progress saved. Total labeled so far: {total_labeled}")

    if is_sharded:
        save_shard(bucket, shard_id, shard_state)
    else:
        save_state(bucket, shard_state)
    logger.info(f"Done! Labeled {total_labeled} samples total")


if __name__ == "__main__":
    main()
