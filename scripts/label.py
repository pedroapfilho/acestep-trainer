#!/usr/bin/env python3
"""Phase 1: Auto-label audio tracks from HF bucket using ACE-Step LLM.

Designed to run on HuggingFace Jobs with GPU. Downloads audio in batches,
runs the DiT audio encoder + LLM labeling, updates dataset.json in the bucket.

Usage (local test):
    uv run python scripts/label.py --bucket pedroapfilho/lofi-tracks --batch-size 50

Usage (HF Job — see scripts/submit_job.py):
    Submitted via submit_job.py which handles cloning, deps, and secrets.
"""

import argparse
import os
import sys
import tempfile

from loguru import logger

# Add ace-step-1.5 to path so we can import its modules
ACE_STEP_DIR = os.path.join(os.path.dirname(__file__), "..", "ace-step-1.5")
sys.path.insert(0, os.path.abspath(ACE_STEP_DIR))

from acestep_trainer.bucket import download_files, list_audio_files
from acestep_trainer.state import (
    DatasetState,
    load_state,
    save_state,
    sync_files_to_state,
)


def init_handlers(model_dir: str | None = None):
    """Initialize ACE-Step DiT and LLM handlers."""
    from acestep.pipeline_ace_step import ACEStepPipeline

    pipe = ACEStepPipeline()

    # Download models if needed (HF Jobs start fresh)
    if model_dir:
        pipe.load_from_directory(model_dir)
    else:
        pipe.load_default()

    return pipe.dit_handler, pipe.llm_handler


def label_batch(
    dit_handler,
    llm_handler,
    state: DatasetState,
    bucket: str,
    batch: list,
    work_dir: str,
) -> int:
    """Download and label a batch of samples. Returns count of newly labeled."""
    from acestep.training.dataset_builder_modules.builder import DatasetBuilder
    from acestep.training.dataset_builder_modules.models import AudioSample

    # Download audio files for this batch
    files_to_download = [s.file for s in batch]
    local_paths = download_files(bucket, files_to_download, work_dir)

    # Build a DatasetBuilder with these samples for labeling
    builder = DatasetBuilder()
    for sample_state, local_path in zip(batch, local_paths):
        builder.samples.append(
            AudioSample(
                audio_path=local_path,
                filename=os.path.basename(local_path),
                is_instrumental=True,
            )
        )

    # Run labeling
    _, status = builder.label_all_samples(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        format_lyrics=False,
        transcribe_lyrics=False,
        skip_metas=False,
        only_unlabeled=False,
    )
    logger.info(f"Labeling result: {status}")

    # Transfer labels back to state
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


def main():
    parser = argparse.ArgumentParser(description="Label audio tracks from HF bucket")
    parser.add_argument("--bucket", required=True, help="HF bucket name (user/repo)")
    parser.add_argument("--batch-size", type=int, default=50, help="Files per batch")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to label (0=all)")
    parser.add_argument("--model-dir", default=None, help="Local model directory")
    parser.add_argument("--save-every", type=int, default=1, help="Save state every N batches")
    args = parser.parse_args()

    logger.info(f"Starting labeling for bucket: {args.bucket}")

    # Load or create state
    state = load_state(args.bucket)

    # Discover audio files in bucket
    logger.info("Scanning bucket for audio files...")
    audio_files = list_audio_files(args.bucket)
    logger.info(f"Found {len(audio_files)} audio files in bucket")

    # Add new files to state
    new_count = sync_files_to_state(args.bucket, state, audio_files)
    if new_count > 0:
        save_state(args.bucket, state)

    # Get unlabeled samples
    unlabeled = state.get_by_status("unlabeled")
    if not unlabeled:
        logger.info("All samples are already labeled!")
        return

    if args.max_samples > 0:
        unlabeled = unlabeled[: args.max_samples]

    logger.info(f"Labeling {len(unlabeled)} unlabeled samples in batches of {args.batch_size}")

    # Initialize models
    logger.info("Initializing ACE-Step models...")
    dit_handler, llm_handler = init_handlers(args.model_dir)

    # Process in batches
    total_labeled = 0
    for batch_idx in range(0, len(unlabeled), args.batch_size):
        batch = unlabeled[batch_idx : batch_idx + args.batch_size]
        batch_num = batch_idx // args.batch_size + 1
        total_batches = (len(unlabeled) + args.batch_size - 1) // args.batch_size

        logger.info(f"Batch {batch_num}/{total_batches}: {len(batch)} samples")

        with tempfile.TemporaryDirectory(prefix="acestep_label_") as work_dir:
            labeled = label_batch(dit_handler, llm_handler, state, args.bucket, batch, work_dir)
            total_labeled += labeled
            logger.info(f"Batch {batch_num}: labeled {labeled}/{len(batch)}")

        # Save state periodically
        if batch_num % args.save_every == 0:
            save_state(args.bucket, state)
            logger.info(f"Progress saved. Total labeled so far: {total_labeled}")

    # Final save
    save_state(args.bucket, state)
    logger.info(f"Done! Labeled {total_labeled} samples total")


if __name__ == "__main__":
    main()
