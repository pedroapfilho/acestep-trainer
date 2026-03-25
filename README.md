# acestep-trainer

Remote training pipeline for [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) LoRA fine-tuning. Runs entirely on [HuggingFace Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs) — no local GPU required.

## How it works

```
HF Bucket (audio)  →  Label  →  Preprocess  →  Train  →  HF Model Repo (LoRA)
                       (LLM)     (VAE)          (LoRA)
```

Three phases, each running as parallel HF Jobs. `dataset.json` in the bucket tracks every sample's status (`unlabeled` → `labeled` → `preprocessed`). Each phase saves progress after every batch — crash-resumable by design.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [hf CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) (for bucket operations)
- HuggingFace Pro account with compute credits
- `HF_TOKEN` environment variable with write access

## Setup

```bash
git clone --recurse-submodules https://github.com/pedroapfilho/acestep-trainer.git
cd acestep-trainer
uv sync
```

## Quick start (full walkthrough)

### 1. Prepare your bucket

Create a HuggingFace Bucket and upload your audio files organized in batch folders:

```
your-bucket/
├── batch-1/
│   ├── track_001.opus
│   ├── track_002.opus
│   └── ...
├── batch-2/
│   └── ...
└── ...
```

Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`

### 2. Scan the bucket

Register all audio files in `dataset.json`:

```bash
uv run acestep-train scan <user/bucket>
uv run acestep-train status <user/bucket>
```

### 3. Label (parallel)

Auto-generate captions, BPM, key, genre, and time signature for each track using the ACE-Step LLM. Use `--parallel` to split across multiple cheap GPUs:

```bash
# 24 parallel shards on T4 ($0.40/hr each) — cheapest option for labeling
uv run python scripts/submit_job.py label \
  --bucket <user/bucket> \
  --parallel 24 \
  --flavor t4-small \
  --timeout 12h
```

The T4 (16GB) is enough for labeling — it only runs the DiT audio encoder (for audio codes) and a 0.6B LLM (for captioning). No need for expensive GPUs here.

Each shard writes to its own `labels_shard_N.json` file. Use `--live` to auto-merge shards every 5 minutes while monitoring:

```bash
uv run acestep-train status <user/bucket> --live
```

Or merge manually at any time:

```bash
uv run acestep-train merge <user/bucket>
```

### 4. Preprocess (parallel)

VAE-encode labeled audio into `.pt` tensors. This needs A100 (80GB) for the VAE encoder:

```bash
# 8 parallel A100 shards
uv run python scripts/submit_job.py preprocess \
  --bucket <user/bucket> \
  --parallel 8 \
  --flavor a100-large \
  --timeout 12h
```

No merge needed for preprocessing — each shard uploads `.pt` files directly to the `tensors/` prefix in the bucket.

### 5. Create a model repo

```bash
python -c "from huggingface_hub import HfApi; HfApi().create_repo('user/model-name', exist_ok=True, private=True)"
```

### 6. Train

Run LoRA fine-tuning on the preprocessed tensors:

```bash
uv run python scripts/submit_job.py train \
  --bucket <user/bucket> \
  --output-repo <user/model-name> \
  --flavor a100-large \
  --timeout 12h \
  --lora-rank 8 \
  --lora-alpha 16 \
  --max-epochs 100
```

Checkpoints are pushed to the model repo every N epochs. The final adapter lands at `<user/model-name>/final/adapter_model.safetensors`.

### 7. Use the LoRA

Load in ACE-Step:

```bash
cd ace-step-1.5
uv run acestep --lora_path <user/model-name>/final
```

Or via the Gradio UI: select your LoRA from the adapter dropdown in the Generation tab.

## MVP training (test the pipeline fast)

You don't need to wait for all samples to be labeled/preprocessed. Run training on whatever tensors exist:

```bash
# Short training run on available tensors (10 epochs)
uv run python scripts/submit_job.py train \
  --bucket <user/bucket> \
  --output-repo <user/model-name> \
  --flavor a100-large \
  --timeout 2h \
  --max-epochs 10
```

Even 500 tensors will produce a usable LoRA — enough to validate the style transfer.

## CLI reference

```bash
# Dataset status (one-shot)
uv run acestep-train status <bucket>

# Live monitoring with auto-merge every 5 minutes
uv run acestep-train status <bucket> --live

# Scan bucket for audio files
uv run acestep-train scan <bucket>

# Merge label shards (auto-detects shard files)
uv run acestep-train merge <bucket>

# Monitor HF Jobs
uv run python scripts/monitor_jobs.py          # live dashboard
uv run python scripts/monitor_jobs.py --once   # one-shot
uv run python scripts/monitor_jobs.py --logs   # with log tails
```

## Job submission reference

All phases support `--dry-run` to preview without submitting.

```bash
uv run python scripts/submit_job.py <phase> [options]
```

| Phase | Key options |
|-------|------------|
| `label` | `--parallel N`, `--max-samples N`, `--batch-size N` |
| `preprocess` | `--parallel N`, `--max-samples N`, `--max-duration N` |
| `train` | `--output-repo`, `--lora-rank`, `--lora-alpha`, `--learning-rate`, `--max-epochs` |

Common options: `--bucket`, `--flavor`, `--timeout`, `--dry-run`

## GPU flavors and costs

| Flavor | VRAM | Cost/hr | Use for |
|--------|------|---------|---------|
| t4-small | 16 GB | $0.40 | Labeling (cheapest) |
| t4-medium | 16 GB | $0.60 | Labeling |
| a10g-small | 24 GB | $1.00 | Labeling (faster) |
| a10g-large | 24 GB | $1.50 | Labeling (faster) |
| a100-large | 80 GB | $2.50 | Preprocessing, training |
| h200 | 141 GB | $5.00 | Large batch training |

## Cost estimates (37k tracks)

| Phase | GPU | Parallelism | Wall time | Total cost |
|-------|-----|-------------|-----------|------------|
| Label | T4-small | 24 shards | ~4h | ~$38 |
| Preprocess | A100 | 8 shards | ~2.5h | ~$50 |
| Train (100 epochs) | A100 | 1 | ~6-12h | ~$15-30 |
| **Total** | | | | **~$103-118** |

## Project structure

```
acestep-trainer/
├── ace-step-1.5/              # git submodule (ACE-Step source)
├── src/acestep_trainer/
│   ├── bucket.py              # HF bucket I/O (hf CLI wrapper)
│   ├── bucket_init.py         # Bucket template/initialization
│   ├── state.py               # dataset.json state machine
│   ├── handler.py             # ACE-Step model initialization
│   └── cli.py                 # CLI (status, scan, merge)
├── scripts/
│   ├── label.py               # Phase 1: auto-labeling (shardable)
│   ├── preprocess.py          # Phase 2: VAE → tensors (shardable)
│   ├── train.py               # Phase 3: LoRA training
│   ├── submit_job.py          # HF Jobs submission with --parallel
│   ├── monitor_jobs.py        # Job monitoring dashboard
│   └── patch_torchaudio.py    # Workaround for torchcodec CUDA mismatch
├── docs/specs/
│   └── 2026-03-24-remote-training-pipeline-design.md
└── pyproject.toml
```

## Bucket structure (after full pipeline)

```
your-bucket/
├── batch-1/                   # Original audio files
│   └── *.opus
├── batch-2/
│   └── *.opus
├── ...
├── dataset.json               # State file (sample metadata + status)
├── labels_shard_0.json        # Temporary shard files (during parallel labeling)
├── labels_shard_1.json
├── ...
└── tensors/                   # Preprocessed tensor files
    ├── manifest.json
    └── *.pt
```

## Troubleshooting

**torchcodec CUDA mismatch**: The `preprocess.py` and `train.py` scripts include `patch_torchaudio.py` which stubs out torchcodec (requires CUDA 13 but HF Jobs have CUDA 12.8). Audio loading falls back to soundfile via ffmpeg.

**safe_path errors**: ACE-Step restricts file paths to be within the project root. Tensor and output directories are created inside `ace-step-1.5/` on the job machine.

**HfFileSystem vs hf CLI**: The Python `huggingface_hub` package (pinned to <1.0 by transformers) doesn't support buckets. All bucket I/O uses the standalone `hf` CLI (installed via `uv tool install`).

**Job crashes**: All phases are crash-resumable. Just resubmit — they skip already-processed samples based on `dataset.json` status.
