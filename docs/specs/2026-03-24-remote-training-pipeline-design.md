# ACE-Step 1.5 Remote Training Pipeline

## Overview

Train ACE-Step 1.5 LoRA adapters on a lo-fi music dataset stored in a HuggingFace Bucket (`pedroapfilho/lofi-tracks`), with all GPU work running on HuggingFace Jobs infrastructure — nothing runs locally except job submission and status checks.

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  HF Bucket          │     │  HF Jobs (GPU)       │     │  HF Model Repo       │
│  pedroapfilho/      │────▶│                      │────▶│  pedroapfilho/       │
│  lofi-tracks        │     │  1. Label (LLM)      │     │  acestep-lofi-lora   │
│                     │◀────│  2. Preprocess (VAE)  │     │                      │
│  ~37k .opus files   │     │  3. Train (LoRA)      │     │  adapter_model.      │
│  dataset.json       │     │                      │     │  safetensors         │
└─────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

## Dataset

- **Source**: `pedroapfilho/lofi-tracks` HuggingFace Bucket
- **Size**: ~39.4 GB, ~37,655 `.opus` files across 158 batch folders
- **State file**: `dataset.json` — lives in the bucket, tracks per-sample status

## Three-Phase Pipeline

### Phase 1: Labeling (`scripts/label.py`)

- Downloads audio in batches to a temp directory
- Runs ACE-Step DiT encoder → audio codes → LLM captioning
- Produces: caption, genre, BPM, key, time signature, language per sample
- Updates `dataset.json` in bucket after each batch (crash-resumable)
- GPU: A10G (24GB) — LLM inference is not VRAM-heavy

### Phase 2: Preprocessing (`scripts/preprocess.py`)

- Downloads labeled audio + runs VAE tiled encoding → `.pt` tensors
- Each tensor contains: `target_latents`, `encoder_hidden_states`, `context_latents`, masks
- Uploads tensors to `tensors/` prefix in the same bucket
- Updates `dataset.json` status to "preprocessed"
- GPU: A100 (80GB) — VAE encoding on long audio is memory-intensive

### Phase 3: Training (`scripts/train.py`)

- Syncs all `.pt` tensors from bucket to local (on job machine)
- Runs ACE-Step LoRA trainer with Lightning Fabric
- Pushes checkpoints to HF model repo every N epochs
- GPU: A100 (80GB) — training with batch_size=1, grad_accum=4

## State Management

`dataset.json` is the single source of truth. Sample status flow:

```
unlabeled → labeled → preprocessed
```

Each phase only processes samples in its input status. If a job crashes, the next run picks up from where it left off — only un-processed samples get worked on.

## Cost Estimate

| Phase        | GPU         | Est. Duration | Est. Cost |
|-------------|-------------|---------------|-----------|
| Label (37k) | A10G 24GB   | 6–12h         | $9–18     |
| Preprocess   | A100 80GB   | 12–24h        | $30–60    |
| Train (100ep)| A100 80GB   | 6–12h         | $15–30    |
| **Total**    |             |               | **$54–108**|

## Project Structure

```
acestep-trainer/
├── ace-step-1.5/              # git submodule
├── src/acestep_trainer/
│   ├── bucket.py              # HfFileSystem bucket helpers
│   ├── state.py               # dataset.json state management
│   └── cli.py                 # Local CLI (status, scan)
├── scripts/
│   ├── label.py               # Phase 1: auto-labeling
│   ├── preprocess.py          # Phase 2: VAE → tensors
│   ├── train.py               # Phase 3: LoRA training
│   └── submit_job.py          # HF Jobs submission helper
└── pyproject.toml             # uv project config
```

## Usage

```bash
# Check dataset state
uv run acestep-train status pedroapfilho/lofi-tracks

# Scan bucket and populate dataset.json
uv run acestep-train scan pedroapfilho/lofi-tracks

# Submit jobs (dry run first)
uv run python scripts/submit_job.py label --bucket pedroapfilho/lofi-tracks --dry-run
uv run python scripts/submit_job.py preprocess --bucket pedroapfilho/lofi-tracks --dry-run
uv run python scripts/submit_job.py train \
    --bucket pedroapfilho/lofi-tracks \
    --output-repo pedroapfilho/acestep-lofi-lora \
    --dry-run

# Small test run (10 samples)
uv run python scripts/submit_job.py label \
    --bucket pedroapfilho/lofi-tracks \
    --max-samples 10 \
    --flavor a10g-small \
    --timeout 1h
```
