"""Cog Predictor for ACE-Step 1.5 with LoRA support.

Deploys to Replicate as frow/lofi.
Weights are downloaded at setup() using pget for fast parallel downloads.
"""

import importlib.machinery
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

# Patch torchcodec (CUDA 13 libnvrtc dep mismatch on all cloud GPU providers)
def _patch_torchcodec():
    def _stub(name):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        m.__loader__ = None
        m.__path__ = []
        m.__package__ = name.rsplit(".", 1)[0] if "." in name else name
        return m

    def _load(uri, *_a, **_k):
        import numpy as np, soundfile as sf, torch
        d, sr = sf.read(uri, dtype="float32", always_2d=True)
        return torch.from_numpy(np.ascontiguousarray(d.T)), sr

    def _save(uri, src, sr, *_a, **_k):
        import numpy as np, soundfile as sf, torch
        d = src.cpu().numpy().T if isinstance(src, torch.Tensor) else np.array(src).T
        sf.write(uri, d, sr)

    ta = _stub("torchaudio._torchcodec")
    ta.load_with_torchcodec = _load
    ta.save_with_torchcodec = _save
    sys.modules["torchaudio._torchcodec"] = ta
    for n in ["torchcodec", "torchcodec.decoders", "torchcodec._internally_replaced_utils"]:
        sys.modules[n] = _stub(n)

_patch_torchcodec()

import torchaudio
from cog import BasePredictor, Input, Path as CogPath

CHECKPOINTS_DIR = "/src/ace-step-1.5/checkpoints"
LORA_DIR = "/src/lora"


class Predictor(BasePredictor):
    def setup(self):
        """Download weights via pget and initialize the model."""
        # Download base model weights (if not already present)
        if not os.path.exists(os.path.join(CHECKPOINTS_DIR, "acestep-v15-turbo")):
            print("Downloading ACE-Step model weights...")
            subprocess.check_call(
                [
                    "pget",
                    "https://huggingface.co/ACE-Step/Ace-Step1.5/resolve/main",
                    CHECKPOINTS_DIR,
                    "-x",
                ],
                close_fds=True,
            )

        # If pget doesn't work with HF URLs, fall back to huggingface_hub
        if not os.path.exists(os.path.join(CHECKPOINTS_DIR, "acestep-v15-turbo")):
            print("Falling back to huggingface_hub download...")
            from huggingface_hub import snapshot_download

            snapshot_download("ACE-Step/Ace-Step1.5", local_dir=CHECKPOINTS_DIR)

        # Download LoRA weights
        if not os.path.exists(os.path.join(LORA_DIR, "adapter_config.json")):
            print("Downloading LoRA weights...")
            from huggingface_hub import snapshot_download

            snapshot_download(
                "pedroapfilho/acestep-lofi-lora",
                local_dir="/src/lora_full",
                allow_patterns="final/adapter/*",
            )
            os.makedirs(LORA_DIR, exist_ok=True)
            src = "/src/lora_full/final/adapter"
            for f in os.listdir(src):
                os.rename(os.path.join(src, f), os.path.join(LORA_DIR, f))

        # Initialize handler
        from acestep.handler import AceStepHandler

        self.handler = AceStepHandler()
        status, ok = self.handler.initialize_service(
            project_root="/src/ace-step-1.5",
            config_path="acestep-v15-turbo",
            device="cuda",
            use_flash_attention=True,
            compile_model=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            quantization=None,
            prefer_source="huggingface",
        )
        if not ok:
            raise RuntimeError(f"Failed to initialize: {status}")

        # Load LoRA
        lora_status = self.handler.load_lora(LORA_DIR)
        print(f"LoRA: {lora_status}")

    def predict(
        self,
        prompt: str = Input(
            description="Text description of the music to generate",
            default="lo-fi hip-hop, laid-back groove, warm analog compression, tape warmth, in Am, at 80 BPM, instrumental only",
        ),
        lyrics: str = Input(
            description="Lyrics for the song. Use '[Instrumental]' for no vocals.",
            default="[Instrumental]",
        ),
        duration: float = Input(
            description="Audio duration in seconds",
            default=60.0,
            ge=10.0,
            le=240.0,
        ),
        seed: int = Input(
            description="Random seed. -1 for random.",
            default=-1,
        ),
        inference_steps: int = Input(
            description="Number of diffusion steps (8 for turbo model)",
            default=8,
            ge=1,
            le=50,
        ),
    ) -> CogPath:
        """Generate music from a text prompt."""
        result = self.handler.generate_music(
            captions=prompt,
            lyrics=lyrics,
            audio_duration=float(duration),
            use_random_seed=seed < 0,
            seed=seed if seed >= 0 else -1,
            inference_steps=inference_steps,
            batch_size=1,
        )

        audios = result.get("audios", [])
        if not audios:
            raise RuntimeError("Generation failed — no audio output")

        audio_data = audios[0]
        tensor = audio_data["tensor"]
        sr = audio_data["sample_rate"]

        # Trim to requested duration
        max_samples = int(duration * sr)
        if tensor.shape[-1] > max_samples:
            tensor = tensor[..., :max_samples]

        out_path = Path(tempfile.mktemp(suffix=".wav"))
        torchaudio.save(str(out_path), tensor, sr)
        return CogPath(out_path)
