"""Cog Predictor for ACE-Step 1.5 with LoRA support.

Deploys to Replicate as a serverless music generation model.
The base model (ACE-Step) is installed in the Docker image at build time.
Model weights and LoRA are downloaded during setup().
"""

import os
import sys
import tempfile
import types
import importlib.machinery
from pathlib import Path

# Patch torchcodec before any torchaudio imports (CUDA 13 dep mismatch)
def _patch_torchcodec():
    def _make_stub(name):
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        mod.__loader__ = None
        mod.__path__ = []
        mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
        return mod

    def _load_sf(uri, *_a, **_kw):
        import numpy as np
        import soundfile as sf
        import torch
        data, sr = sf.read(uri, dtype="float32", always_2d=True)
        return torch.from_numpy(np.ascontiguousarray(data.T)), sr

    def _save_sf(uri, src, sample_rate, *_a, **_kw):
        import numpy as np
        import soundfile as sf
        import torch
        if isinstance(src, torch.Tensor):
            data = src.cpu().numpy().T
        else:
            data = np.array(src).T
        sf.write(uri, data, sample_rate)

    ta = _make_stub("torchaudio._torchcodec")
    ta.load_with_torchcodec = _load_sf
    ta.save_with_torchcodec = _save_sf
    sys.modules["torchaudio._torchcodec"] = ta
    for name in ["torchcodec", "torchcodec.decoders", "torchcodec._internally_replaced_utils"]:
        sys.modules[name] = _make_stub(name)

_patch_torchcodec()

import torchaudio
from cog import BasePredictor, Input, Path as CogPath


class Predictor(BasePredictor):
    def setup(self):
        """Load model weights (baked into image) and initialize handler."""
        from acestep.handler import AceStepHandler

        project_root = "/src/ace-step-1.5"

        self.handler = AceStepHandler()
        status, ok = self.handler.initialize_service(
            project_root=project_root,
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

        # Load LoRA (baked into image at /src/lora)
        lora_status = self.handler.load_lora("/src/lora")
        print(f"LoRA loaded: {lora_status}")

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
