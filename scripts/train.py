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

from __future__ import annotations

# Patch torchaudio before any imports that trigger it
# (torchcodec has a CUDA 13 dep mismatch on HF Jobs)
import patch_torchaudio

patch_torchaudio.patch()

import argparse
import os
import tempfile

from huggingface_hub import HfApi
from loguru import logger

from acestep_trainer.bucket import _hf_url
from acestep_trainer.bucket import _run_hf
from acestep_trainer.handler import ensure_sys_path
from acestep_trainer.handler import init_dit_handler


def sync_tensors(bucket: str, dest_dir: str) -> str:
    """Download all tensor files from the bucket to local dir via hf CLI."""
    _run_hf("buckets", "sync", _hf_url(bucket, "tensors"), dest_dir)
    count = sum(1 for f in os.listdir(dest_dir) if f.endswith((".pt", ".json")))
    logger.info(f"Synced {count} files to {dest_dir}")
    return dest_dir


def push_checkpoint(
    checkpoint_dir: str,
    repo_id: str,
    epoch: int,
    is_final: bool = False,
) -> None:
    """Push a checkpoint to a HuggingFace model repo."""
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, private=True)

    subfolder = "final" if is_final else f"checkpoints/epoch_{epoch}"
    api.upload_folder(
        repo_id=repo_id,
        folder_path=checkpoint_dir,
        path_in_repo=subfolder,
        commit_message=f"{'Final' if is_final else f'Epoch {epoch}'} checkpoint",
    )
    logger.info(f"Pushed checkpoint to {repo_id}/{subfolder}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA from preprocessed tensors")
    parser.add_argument("--bucket", required=True, help="HF bucket with tensors/")
    parser.add_argument("--output-repo", required=True, help="HF model repo for checkpoints")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10, help="Save every N epochs")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--resume-from", default=None, help="Checkpoint dir to resume from")
    parsed = parser.parse_args()

    bucket: str = parsed.bucket
    output_repo: str = parsed.output_repo
    lora_rank: int = parsed.lora_rank
    lora_alpha: int = parsed.lora_alpha
    lora_dropout: float = parsed.lora_dropout
    learning_rate: float = parsed.learning_rate
    batch_size: int = parsed.batch_size
    gradient_accumulation: int = parsed.gradient_accumulation
    max_epochs: int = parsed.max_epochs
    save_every: int = parsed.save_every
    warmup_steps: int = parsed.warmup_steps
    resume_from: str | None = parsed.resume_from

    logger.info(f"Starting LoRA training from bucket: {bucket}")

    tensor_dir = tempfile.mkdtemp(prefix="acestep_tensors_")
    logger.info("Syncing tensor files from bucket...")
    sync_tensors(bucket, tensor_dir)

    output_dir = tempfile.mkdtemp(prefix="acestep_output_")

    logger.info("Initializing ACE-Step DiT handler...")
    ensure_sys_path()
    dit_handler = init_dit_handler(quantization=None)

    from acestep.training.configs import LoRAConfig  # type: ignore[import-untyped]
    from acestep.training.configs import TrainingConfig  # type: ignore[import-untyped]
    from acestep.training.trainer import LoRATrainer  # type: ignore[import-untyped]

    lora_config = LoRAConfig(r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

    training_config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        max_epochs=max_epochs,
        save_every_n_epochs=save_every,
        warmup_steps=warmup_steps,
        output_dir=output_dir,
    )

    trainer = LoRATrainer(
        dit_handler=dit_handler,
        lora_config=lora_config,
        training_config=training_config,
    )

    logger.info(
        f"Training: rank={lora_rank}, alpha={lora_alpha}, "
        f"lr={learning_rate}, epochs={max_epochs}, "
        f"batch={batch_size}, grad_accum={gradient_accumulation}"
    )

    last_pushed_epoch = -1
    for step, loss, status in trainer.train_from_preprocessed(
        tensor_dir=tensor_dir,
        training_state={"should_stop": False},
        resume_from=resume_from,
    ):
        logger.info(f"Step {step} | Loss: {loss:.6f} | {status}")

        if "saved checkpoint" in status.lower():
            checkpoints_path = os.path.join(output_dir, "checkpoints")
            checkpoint_dirs = sorted(
                d
                for d in os.listdir(checkpoints_path)
                if os.path.isdir(os.path.join(checkpoints_path, d))
            )
            if checkpoint_dirs:
                latest = checkpoint_dirs[-1]
                epoch_num = int(latest.split("_")[-1]) if "_" in latest else 0
                if epoch_num > last_pushed_epoch:
                    push_checkpoint(
                        os.path.join(checkpoints_path, latest),
                        output_repo,
                        epoch_num,
                    )
                    last_pushed_epoch = epoch_num

    final_dir = os.path.join(output_dir, "final")
    if os.path.exists(final_dir):
        push_checkpoint(final_dir, output_repo, 0, is_final=True)
        logger.info(f"Final model pushed to {output_repo}/final")
    else:
        logger.warning("No final/ directory found — check training output")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
