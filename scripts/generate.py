#!/usr/bin/env python3
"""Generate audio using ACE-Step with a trained LoRA adapter.

Runs on HuggingFace Jobs with GPU. Loads the base model + LoRA,
generates audio from a prompt, and uploads the result to the bucket.

Usage (HF Job):
    python scripts/generate.py \
        --lora-repo pedroapfilho/acestep-lofi-lora/final \
        --prompt "lo-fi hip-hop, laid-back groove, tape warmth, instrumental only" \
        --output-bucket pedroapfilho/lofi-tracks

Usage (local via submit):
    uv run python scripts/submit_job.py generate \
        --lora-repo pedroapfilho/acestep-lofi-lora/final \
        --prompt "lo-fi hip-hop, warm analog, in Dm, at 80 BPM, instrumental only"
"""

from __future__ import annotations

import patch_torchaudio

patch_torchaudio.patch()

import argparse
import os

import torchaudio  # type: ignore[import-untyped]
from loguru import logger

from acestep_trainer.bucket import upload_file
from acestep_trainer.handler import ensure_sys_path
from acestep_trainer.handler import init_dit_handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio with ACE-Step + LoRA")
    parser.add_argument(
        "--lora-repo", required=True, help="HF model repo with LoRA (user/repo/subfolder)"
    )
    parser.add_argument("--prompt", required=True, help="Caption/prompt for generation")
    parser.add_argument("--lyrics", default="[Instrumental]", help="Lyrics (default: instrumental)")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--output-bucket", default="", help="Upload result to this HF bucket")
    parser.add_argument("--output-dir", default="./output", help="Local output directory")
    parser.add_argument("--num-tracks", type=int, default=1, help="Number of tracks to generate")
    parsed = parser.parse_args()

    lora_repo: str = parsed.lora_repo
    prompt: str = parsed.prompt
    lyrics: str = parsed.lyrics
    duration: int = parsed.duration
    seed: int = parsed.seed
    output_bucket: str = parsed.output_bucket
    output_dir: str = parsed.output_dir
    num_tracks: int = parsed.num_tracks

    logger.info(f"Generating {num_tracks} track(s) with LoRA: {lora_repo}")
    logger.info(f"Prompt: {prompt}")

    # Initialize model
    ensure_sys_path()
    dit_handler = init_dit_handler()

    # Download and load LoRA
    # lora_repo can be "user/repo" or "user/repo/subfolder"
    logger.info(f"Loading LoRA from {lora_repo}...")
    import tempfile

    from huggingface_hub import snapshot_download

    parts = lora_repo.split("/")
    if len(parts) > 2:
        repo_id = "/".join(parts[:2])
        subfolder = "/".join(parts[2:])
        lora_local = snapshot_download(
            repo_id, local_dir=tempfile.mkdtemp(prefix="lora_"), allow_patterns=f"{subfolder}/*"
        )
        lora_local = os.path.join(lora_local, subfolder)
    else:
        lora_local = snapshot_download(lora_repo, local_dir=tempfile.mkdtemp(prefix="lora_"))

    status = dit_handler.load_lora(lora_local)
    logger.info(f"LoRA: {status}")

    # Generate
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_tracks):
        logger.info(f"Generating track {i + 1}/{num_tracks}...")

        result = dit_handler.generate_music(
            captions=prompt,
            lyrics=lyrics,
            audio_duration=float(duration),
            use_random_seed=seed < 0,
            seed=seed if seed >= 0 else -1,
        )

        audios = result.get("audios", [])
        if not audios:
            logger.warning(f"Track {i + 1}: no audio generated")
            continue

        audio_data = audios[0]
        tensor = audio_data["tensor"]
        sr = audio_data["sample_rate"]

        output_path = os.path.join(output_dir, f"generated_{i:03d}.wav")
        torchaudio.save(output_path, tensor, sr)
        logger.info(f"Saved: {output_path} ({tensor.shape[1] / sr:.1f}s, {sr}Hz)")

        if output_bucket:
            remote_path = f"generated/generated_{i:03d}.wav"
            upload_file(output_bucket, output_path, remote_path)
            logger.info(f"Uploaded to {output_bucket}/{remote_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
