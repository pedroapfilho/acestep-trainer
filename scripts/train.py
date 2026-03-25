#!/usr/bin/env python3
"""Phase 3: LoRA training on preprocessed tensors from HF bucket.

Syncs tensor files from the bucket, runs the ACE-Step LoRA trainer, and
pushes checkpoints to a HuggingFace model repo.

Requires GPU — designed to run on HuggingFace Jobs.

Usage:
    uv run python scripts/train.py \
        --bucket pedroapfilho/lofi-tracks \
        --output-repo pedroapfilho/acestep-lofi-lora \
        --max-epochs 100
"""

import argparse
import os
import tempfile

from huggingface_hub import HfApi
from loguru import logger

from acestep_trainer.handler import ensure_sys_path, init_dit_handler


def sync_tensors(bucket: str, dest_dir: str) -> str:
    """Download all tensor files from the bucket to local dir."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    tensor_prefix = f"buckets/{bucket}/tensors"

    if not fs.exists(tensor_prefix):
        raise FileNotFoundError(f"No tensors/ directory found in bucket {bucket}")

    # Download manifest and all .pt files
    files = fs.ls(tensor_prefix, detail=False)
    count = 0
    for f in files:
        name = os.path.basename(f)
        local_path = os.path.join(dest_dir, name)
        if not os.path.exists(local_path):
            fs.get(f, local_path)
            count += 1

    logger.info(f"Synced {count} tensor files to {dest_dir}")
    return dest_dir


def push_checkpoint(
    checkpoint_dir: str,
    repo_id: str,
    epoch: int,
    is_final: bool = False,
) -> None:
    """Push a checkpoint to a HuggingFace model repo."""
    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id, exist_ok=True, private=True)

    subfolder = "final" if is_final else f"checkpoints/epoch_{epoch}"
    api.upload_folder(
        repo_id=repo_id,
        folder_path=checkpoint_dir,
        path_in_repo=subfolder,
        commit_message=f"{'Final' if is_final else f'Epoch {epoch}'} checkpoint",
    )
    logger.info(f"Pushed checkpoint to {repo_id}/{subfolder}")


def main():
    parser = argparse.ArgumentParser(description="Train LoRA from preprocessed tensors")
    parser.add_argument("--bucket", required=True, help="HF bucket with tensors/")
    parser.add_argument("--output-repo", required=True, help="HF model repo for checkpoints")

    # LoRA config
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")

    # Training config
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10, help="Save every N epochs")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--resume-from", default=None, help="Checkpoint dir to resume from")

    args = parser.parse_args()

    logger.info(f"Starting LoRA training from bucket: {args.bucket}")

    # Sync tensors from bucket to local temp dir
    tensor_dir = tempfile.mkdtemp(prefix="acestep_tensors_")
    logger.info("Syncing tensor files from bucket...")
    sync_tensors(args.bucket, tensor_dir)

    # Output dir for checkpoints (local, pushed to HF after each save)
    output_dir = tempfile.mkdtemp(prefix="acestep_output_")

    # Initialize model (downloads from HF on first use, no quantization for LoRA)
    logger.info("Initializing ACE-Step DiT handler...")
    ensure_sys_path()
    dit_handler = init_dit_handler(quantization=None)

    # Configure training
    from acestep.training.configs import LoRAConfig, TrainingConfig
    from acestep.training.trainer import LoRATrainer

    lora_config = LoRAConfig(
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_epochs=args.max_epochs,
        save_every_n_epochs=args.save_every,
        warmup_steps=args.warmup_steps,
        output_dir=output_dir,
    )

    trainer = LoRATrainer(
        dit_handler=dit_handler,
        lora_config=lora_config,
        training_config=training_config,
    )

    # Train
    logger.info(
        f"Training: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"lr={args.learning_rate}, epochs={args.max_epochs}, "
        f"batch={args.batch_size}, grad_accum={args.gradient_accumulation}"
    )

    last_pushed_epoch = -1
    for step, loss, status in trainer.train_from_preprocessed(
        tensor_dir=tensor_dir,
        training_state={"should_stop": False},
        resume_from=args.resume_from,
    ):
        logger.info(f"Step {step} | Loss: {loss:.6f} | {status}")

        # Push checkpoint when a new epoch save occurs
        if "saved checkpoint" in status.lower():
            # Extract epoch from the checkpoint directory
            checkpoint_dirs = sorted(
                [
                    d
                    for d in os.listdir(os.path.join(output_dir, "checkpoints"))
                    if os.path.isdir(os.path.join(output_dir, "checkpoints", d))
                ]
            )
            if checkpoint_dirs:
                latest = checkpoint_dirs[-1]
                epoch_num = int(latest.split("_")[-1]) if "_" in latest else 0
                if epoch_num > last_pushed_epoch:
                    push_checkpoint(
                        os.path.join(output_dir, "checkpoints", latest),
                        args.output_repo,
                        epoch_num,
                    )
                    last_pushed_epoch = epoch_num

    # Push final checkpoint
    final_dir = os.path.join(output_dir, "final")
    if os.path.exists(final_dir):
        push_checkpoint(final_dir, args.output_repo, 0, is_final=True)
        logger.info(f"Final model pushed to {args.output_repo}/final")
    else:
        logger.warning("No final/ directory found — check training output")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
