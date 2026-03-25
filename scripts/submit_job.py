#!/usr/bin/env python3
"""Submit training pipeline phases as HuggingFace Jobs.

Each phase (label, preprocess, train) runs on HF GPU infrastructure.
The job clones both this repo and ace-step-1.5, installs deps, and runs
the appropriate script.

Usage:
    # Label all tracks (needs GPU for audio encoding)
    uv run python scripts/submit_job.py label --bucket pedroapfilho/lofi-tracks

    # Preprocess labeled tracks to tensors
    uv run python scripts/submit_job.py preprocess --bucket pedroapfilho/lofi-tracks

    # Train LoRA
    uv run python scripts/submit_job.py train \
        --bucket pedroapfilho/lofi-tracks \
        --output-repo pedroapfilho/acestep-lofi-lora

    # Small test run (1 batch)
    uv run python scripts/submit_job.py label \
        --bucket pedroapfilho/lofi-tracks \
        --max-samples 10 \
        --flavor a10g-small \
        --timeout 1h
"""

import argparse
import os

TRAINER_REPO = "https://github.com/pedroapfilho/acestep-trainer.git"
ACESTEP_REPO = "https://github.com/ace-step/ACE-Step-1.5.git"

# GPU flavors and their approximate costs
FLAVORS = {
    "t4-small": {"vram": "16GB", "cost_hr": 0.40},
    "t4-medium": {"vram": "16GB", "cost_hr": 0.60},
    "l4": {"vram": "24GB", "cost_hr": 0.80},
    "l40s": {"vram": "48GB", "cost_hr": 1.80},
    "a10g-small": {"vram": "24GB", "cost_hr": 1.00},
    "a10g-large": {"vram": "24GB", "cost_hr": 1.50},
    "a100-large": {"vram": "80GB", "cost_hr": 2.50},
    "h200": {"vram": "141GB", "cost_hr": 5.00},
}

DEFAULT_FLAVORS = {
    "label": "a10g-large",
    "preprocess": "a100-large",
    "train": "a100-large",
}

DEFAULT_TIMEOUTS = {
    "label": "12h",
    "preprocess": "24h",
    "train": "12h",
}


def build_setup_commands() -> str:
    """Commands to set up the environment inside the HF Job."""
    return " && ".join(
        [
            # Install system deps (git not present in pytorch docker image)
            "apt-get update -qq && apt-get install -y -qq git ffmpeg libsndfile1 > /dev/null",
            # Install uv (fast Python package manager)
            "pip install -q uv",
            # Install standalone hf CLI (Rust binary, avoids Python version conflicts)
            "uv tool install 'huggingface_hub[hf_xet,cli]'",
            "export PATH=$HOME/.local/bin:$PATH",
            # Clone repos
            f"git clone {ACESTEP_REPO} /workspace/ace-step-1.5",
            f"git clone {TRAINER_REPO} /workspace/acestep-trainer",
            # Install ace-step with uv (handles local nano-vllm source)
            "cd /workspace/ace-step-1.5",
            "uv pip install --system -e .",
            # Install trainer deps
            "cd /workspace/acestep-trainer",
            "uv pip install --system -e .",
            # Remove torchcodec (broken CUDA 13 dep) — torchaudio+ffmpeg handles audio
            "pip uninstall -y torchcodec 2>/dev/null || true",
        ]
    )


def build_label_command(args: argparse.Namespace, shard_id: int = -1, num_shards: int = 1) -> str:
    """Build the labeling command."""
    cmd = f"python /workspace/acestep-trainer/scripts/label.py --bucket {args.bucket}"
    if args.max_samples:
        cmd += f" --max-samples {args.max_samples}"
    if args.batch_size:
        cmd += f" --batch-size {args.batch_size}"
    if shard_id >= 0:
        cmd += f" --shard-id {shard_id} --num-shards {num_shards}"
    return cmd


def build_preprocess_command(args: argparse.Namespace) -> str:
    """Build the preprocessing command."""
    cmd = f"python /workspace/acestep-trainer/scripts/preprocess.py --bucket {args.bucket}"
    if args.max_samples:
        cmd += f" --max-samples {args.max_samples}"
    if args.batch_size:
        cmd += f" --batch-size {args.batch_size}"
    if args.max_duration:
        cmd += f" --max-duration {args.max_duration}"
    return cmd


def build_train_command(args: argparse.Namespace) -> str:
    """Build the training command."""
    cmd = (
        f"python /workspace/acestep-trainer/scripts/train.py"
        f" --bucket {args.bucket}"
        f" --output-repo {args.output_repo}"
        f" --lora-rank {args.lora_rank}"
        f" --lora-alpha {args.lora_alpha}"
        f" --learning-rate {args.learning_rate}"
        f" --max-epochs {args.max_epochs}"
        f" --batch-size {args.batch_size}"
        f" --gradient-accumulation {args.gradient_accumulation}"
    )
    return cmd


def submit(phase: str, command: str, flavor: str, timeout: str, dry_run: bool = False):
    """Submit a job to HuggingFace."""
    setup = build_setup_commands()
    full_command = f'bash -c "{setup} && {command}"'

    flavor_info = FLAVORS.get(flavor, {})
    vram = flavor_info.get("vram", "?")
    cost = flavor_info.get("cost_hr", 0)

    print(f"\n{'=' * 60}")
    print(f"Phase: {phase}")
    print(f"Flavor: {flavor} ({vram} VRAM, ~${cost}/hr)")
    print(f"Timeout: {timeout}")
    print(f"Command: {command}")
    print(f"{'=' * 60}\n")

    if dry_run:
        print("[DRY RUN] Would submit the following job:")
        print("  Image: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
        print(f"  Flavor: {flavor}")
        print(f"  Timeout: {timeout}")
        print(f"  Command: {full_command}")
        return

    from huggingface_hub import run_job

    job = run_job(
        image="pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel",
        command=["bash", "-c", f"{setup} && {command}"],
        flavor=flavor,  # type: ignore[arg-type]  # HF SDK accepts string flavors
        timeout=timeout,
        secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
    )
    print(f"Job submitted: {job}")


def main():
    parser = argparse.ArgumentParser(description="Submit HF Jobs for ACE-Step training")
    subparsers = parser.add_subparsers(dest="phase", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--bucket", required=True, help="HF bucket name")
    common.add_argument("--flavor", default=None, help="GPU flavor (e.g. a100-large)")
    common.add_argument("--timeout", default=None, help="Job timeout (e.g. 12h)")
    common.add_argument("--dry-run", action="store_true", help="Print command without submitting")

    # Label
    label_parser = subparsers.add_parser("label", parents=[common])
    label_parser.add_argument("--max-samples", type=int, default=0)
    label_parser.add_argument("--batch-size", type=int, default=50)
    label_parser.add_argument(
        "--parallel", type=int, default=1, help="Number of parallel jobs (shards)"
    )

    # Preprocess
    pre_parser = subparsers.add_parser("preprocess", parents=[common])
    pre_parser.add_argument("--max-samples", type=int, default=0)
    pre_parser.add_argument("--batch-size", type=int, default=20)
    pre_parser.add_argument("--max-duration", type=float, default=240.0)

    # Train
    train_parser = subparsers.add_parser("train", parents=[common])
    train_parser.add_argument("--output-repo", required=True, help="HF model repo")
    train_parser.add_argument("--lora-rank", type=int, default=8)
    train_parser.add_argument("--lora-alpha", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--max-epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--gradient-accumulation", type=int, default=4)

    args = parser.parse_args()

    flavor = args.flavor or DEFAULT_FLAVORS[args.phase]
    timeout = args.timeout or DEFAULT_TIMEOUTS[args.phase]

    if args.phase == "label":
        parallel = getattr(args, "parallel", 1)
        if parallel > 1:
            print(f"Submitting {parallel} parallel labeling jobs...")
            merge_cmd = f"uv run acestep-train merge {args.bucket} --num-shards {parallel}"
            print(f"After all jobs complete, run: {merge_cmd}\n")
            for shard_id in range(parallel):
                command = build_label_command(args, shard_id=shard_id, num_shards=parallel)
                phase_name = f"label (shard {shard_id}/{parallel})"
                submit(phase_name, command, flavor, timeout, args.dry_run)
            return
        command = build_label_command(args)
    elif args.phase == "preprocess":
        command = build_preprocess_command(args)
    elif args.phase == "train":
        command = build_train_command(args)
    else:
        parser.error(f"Unknown phase: {args.phase}")

    submit(args.phase, command, flavor, timeout, args.dry_run)


if __name__ == "__main__":
    main()
