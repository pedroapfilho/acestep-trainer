# acestep-trainer

Remote training pipeline for [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) LoRA fine-tuning. Runs entirely on [HuggingFace Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs) — no local GPU required.

## How it works

The pipeline has three phases, each running as a separate HF Job:

```
HF Bucket (audio)  →  Phase 1: Label  →  Phase 2: Preprocess  →  Phase 3: Train  →  HF Model Repo (LoRA)
                       (LLM captions)     (VAE → tensors)         (LoRA training)
```

**`dataset.json`** lives in the HF bucket and tracks every sample's status (`unlabeled` → `labeled` → `preprocessed`). Each phase saves progress after every batch, so jobs are crash-resumable — just resubmit and they pick up where they left off.

### Phase 1: Labeling

Downloads audio in batches, runs ACE-Step's DiT encoder + 5Hz LLM to generate captions, BPM, key, genre, and time signature for each track.

### Phase 2: Preprocessing

Takes labeled audio and encodes it through the VAE into `.pt` tensor files (target latents, text embeddings, context latents). These are what the training loop consumes.

### Phase 3: Training

Runs LoRA fine-tuning on the preprocessed tensors using ACE-Step's flow matching trainer. Pushes checkpoints to a HuggingFace model repo.

## Setup

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/pedroapfilho/acestep-trainer.git
cd acestep-trainer

# Install deps (requires uv)
uv sync

# Ensure HF_TOKEN is set (needs write access)
echo $HF_TOKEN
```

## Usage

### Check dataset status

```bash
uv run acestep-train status <bucket>
uv run acestep-train scan <bucket>
```

### Submit jobs

```bash
# Dry run (shows the command without submitting)
uv run python scripts/submit_job.py label --bucket <bucket> --dry-run

# Label all tracks
uv run python scripts/submit_job.py label --bucket <bucket> --flavor a10g-large --timeout 24h

# Preprocess labeled tracks to tensors
uv run python scripts/submit_job.py preprocess --bucket <bucket> --flavor a100-large --timeout 24h

# Train LoRA
uv run python scripts/submit_job.py train \
  --bucket <bucket> \
  --output-repo <user/model-repo> \
  --flavor a100-large \
  --timeout 12h \
  --lora-rank 8 \
  --lora-alpha 16 \
  --max-epochs 100
```

### Test with a small batch first

```bash
uv run python scripts/submit_job.py label \
  --bucket <bucket> \
  --max-samples 10 \
  --flavor a10g-small \
  --timeout 1h
```

## Project structure

```
acestep-trainer/
├── ace-step-1.5/              # git submodule (ACE-Step source)
├── src/acestep_trainer/
│   ├── bucket.py              # HF bucket I/O (uses hf CLI)
│   ├── state.py               # dataset.json state machine
│   ├── handler.py             # ACE-Step model initialization
│   └── cli.py                 # Local CLI (status, scan)
├── scripts/
│   ├── label.py               # Phase 1: auto-labeling
│   ├── preprocess.py          # Phase 2: VAE → tensors
│   ├── train.py               # Phase 3: LoRA training
│   └── submit_job.py          # HF Jobs submission helper
└── pyproject.toml
```

## GPU flavors and costs

| Flavor | VRAM | Cost/hr | Use for |
|--------|------|---------|---------|
| a10g-small | 24 GB | $1.00 | Testing |
| a10g-large | 24 GB | $1.50 | Labeling |
| a100-large | 80 GB | $2.50 | Preprocessing, training |
| h200 | 141 GB | $5.00 | Large batch training |

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [hf CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) (installed locally for bucket operations)
- HuggingFace Pro account with compute credits (for running jobs)
- `HF_TOKEN` environment variable with write access
