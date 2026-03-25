#!/usr/bin/env python3
"""Phase 2: Preprocess labeled audio to .pt tensors for training.

Downloads labeled audio from the HF bucket, runs VAE encoding + text encoding
via ACE-Step, uploads tensor files back to the bucket under tensors/ prefix.

Requires GPU — designed to run on HuggingFace Jobs.

Usage:
    uv run python scripts/preprocess.py --bucket pedroapfilho/lofi-tracks --batch-size 20
"""

import argparse
import os
import tempfile

from loguru import logger

from acestep_trainer.bucket import download_files, upload_directory
from acestep_trainer.handler import init_dit_handler, ensure_sys_path
from acestep_trainer.state import load_state, save_state


def preprocess_batch(
    dit_handler,
    state,
    bucket: str,
    batch: list,
    work_dir: str,
    tensor_dir: str,
    max_duration: float,
) -> int:
    """Preprocess a batch of labeled samples. Returns count of tensors created."""
    from acestep.training.dataset_builder_modules.builder import DatasetBuilder
    from acestep.training.dataset_builder_modules.models import AudioSample

    # Download audio for batch
    files_to_download = [s.file for s in batch]
    local_paths = download_files(bucket, files_to_download, work_dir)

    # Build DatasetBuilder with labeled metadata from state
    builder = DatasetBuilder()
    builder.metadata.custom_tag = state.custom_tag
    builder.metadata.tag_position = state.tag_position
    builder.metadata.genre_ratio = state.genre_ratio
    builder.metadata.all_instrumental = state.all_instrumental

    for sample_state, local_path in zip(batch, local_paths):
        builder.samples.append(
            AudioSample(
                audio_path=local_path,
                filename=os.path.basename(local_path),
                caption=sample_state.caption,
                genre=sample_state.genre,
                lyrics=sample_state.lyrics,
                bpm=sample_state.bpm,
                keyscale=sample_state.keyscale,
                timesignature=sample_state.timesignature,
                language=sample_state.language,
                is_instrumental=sample_state.is_instrumental,
                labeled=True,
            )
        )

    # Run preprocessing (VAE encode + text encode)
    output_paths, status = builder.preprocess_to_tensors(
        dit_handler=dit_handler,
        output_dir=tensor_dir,
        max_duration=max_duration,
        preprocess_mode="lora",
        skip_existing=True,
    )
    logger.info(f"Preprocess result: {status}")

    # Mark samples as preprocessed in state
    preprocessed_count = 0
    for sample_state, output_path in zip(batch, output_paths):
        if output_path:
            tensor_name = os.path.basename(output_path)
            sample_state_ref = next(
                (s for s in state.samples if s.file == sample_state.file), None
            )
            if sample_state_ref:
                state.mark_preprocessed(sample_state.file, f"tensors/{tensor_name}")
                preprocessed_count += 1

    return preprocessed_count


def main():
    parser = argparse.ArgumentParser(description="Preprocess labeled audio to tensors")
    parser.add_argument("--bucket", required=True, help="HF bucket name (user/repo)")
    parser.add_argument("--batch-size", type=int, default=20, help="Files per batch")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples (0=all)")
    parser.add_argument("--max-duration", type=float, default=240.0, help="Max audio duration (s)")
    parser.add_argument("--save-every", type=int, default=1, help="Save state every N batches")
    args = parser.parse_args()

    logger.info(f"Starting preprocessing for bucket: {args.bucket}")

    # Load state
    state = load_state(args.bucket)

    # Get labeled (but not yet preprocessed) samples
    labeled = state.get_by_status("labeled")
    if not labeled:
        logger.info("No labeled samples to preprocess!")
        return

    if args.max_samples > 0:
        labeled = labeled[: args.max_samples]

    logger.info(f"Preprocessing {len(labeled)} labeled samples")

    # Initialize models (downloads from HF on first use)
    logger.info("Initializing ACE-Step DiT handler...")
    ensure_sys_path()
    dit_handler = init_dit_handler()

    # Process in batches
    total_preprocessed = 0
    for batch_idx in range(0, len(labeled), args.batch_size):
        batch = labeled[batch_idx : batch_idx + args.batch_size]
        batch_num = batch_idx // args.batch_size + 1
        total_batches = (len(labeled) + args.batch_size - 1) // args.batch_size

        logger.info(f"Batch {batch_num}/{total_batches}: {len(batch)} samples")

        with tempfile.TemporaryDirectory(prefix="acestep_pre_") as work_dir:
            tensor_dir = os.path.join(work_dir, "tensors")
            os.makedirs(tensor_dir, exist_ok=True)

            count = preprocess_batch(
                dit_handler, state, args.bucket, batch, work_dir, tensor_dir, args.max_duration
            )
            total_preprocessed += count

            # Upload tensors to bucket
            if count > 0:
                upload_directory(args.bucket, tensor_dir, "tensors")
                logger.info(f"Batch {batch_num}: uploaded {count} tensor files")

        # Save state periodically
        if batch_num % args.save_every == 0:
            save_state(args.bucket, state)
            logger.info(f"Progress saved. Total preprocessed: {total_preprocessed}")

    # Final save
    save_state(args.bucket, state)
    logger.info(f"Done! Preprocessed {total_preprocessed} samples total")


if __name__ == "__main__":
    main()
