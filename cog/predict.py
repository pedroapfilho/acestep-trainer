"""Cog Predictor for ACE-Step 1.5 with LoRA support.

Deploys to Replicate as frow/lofi.
Weights are downloaded at setup() via huggingface_hub.
"""

import importlib.machinery
import os
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
        """Download weights and initialize the model."""
        from huggingface_hub import snapshot_download

        # Download base model weights
        if not os.path.exists(os.path.join(CHECKPOINTS_DIR, "acestep-v15-turbo")):
            print("Downloading ACE-Step model weights...", flush=True)
            snapshot_download("ACE-Step/Ace-Step1.5", local_dir=CHECKPOINTS_DIR)
            print("Model weights downloaded.", flush=True)

        # Download LoRA weights
        if not os.path.exists(os.path.join(LORA_DIR, "adapter_config.json")):
            print("Downloading LoRA weights...", flush=True)
            snapshot_download(
                "pedroapfilho/acestep-lofi-lora",
                local_dir="/src/lora_full",
                allow_patterns="final/adapter/*",
            )
            os.makedirs(LORA_DIR, exist_ok=True)
            src = "/src/lora_full/final/adapter"
            for f in os.listdir(src):
                os.rename(os.path.join(src, f), os.path.join(LORA_DIR, f))
            print("LoRA weights downloaded.", flush=True)

        # Initialize handler
        print("Initializing model...", flush=True)
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
        print("Model initialized.", flush=True)

        # Load LoRA
        lora_status = self.handler.load_lora(LORA_DIR)
        print(f"LoRA: {lora_status}", flush=True)

    def predict(
        self,
        prompt: str = Input(
            description="Text description of the music to generate",
            default="lo-fi hip-hop, laid-back groove, warm analog compression, tape warmth",
        ),
        bpm: int = Input(
            description="Beats per minute. 0 for auto-detection from prompt.",
            default=80,
            ge=0,
            le=240,
        ),
        key_scale: str = Input(
            description="Musical key (e.g. 'Am', 'C Major', 'Dm'). Leave empty for auto.",
            default="Am",
        ),
        time_signature: str = Input(
            description="Time signature (e.g. '4/4', '3/4'). Leave empty for auto.",
            default="4/4",
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
            description="Number of diffusion steps. More steps = higher quality but slower. 8-20 recommended for turbo.",
            default=16,
            ge=4,
            le=50,
        ),
        lora_scale: float = Input(
            description="LoRA influence strength. Lower values blend more with the base model. 0 disables LoRA.",
            default=0.8,
            ge=0.0,
            le=1.0,
        ),
        output_format: str = Input(
            description="Output audio format.",
            default="mp3",
            choices=["wav", "mp3", "flac", "ogg"],
        ),
    ) -> CogPath:
        """Generate music from a text prompt."""
        # Apply LoRA scale
        if lora_scale > 0:
            self.handler.use_lora = True
            self.handler.set_lora_scale(lora_scale)
        else:
            self.handler.use_lora = False

        result = self.handler.generate_music(
            captions=prompt,
            lyrics="[Instrumental]",
            bpm=bpm if bpm > 0 else None,
            key_scale=key_scale,
            time_signature=time_signature,
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

        # Cosine fade-out over the last 5 seconds for a smooth ending
        import torch

        fade_seconds = min(5.0, duration * 0.15)
        fade_samples = int(fade_seconds * sr)
        if fade_samples > 0 and tensor.shape[-1] > fade_samples:
            fade_curve = torch.cos(
                torch.linspace(0, torch.pi / 2, fade_samples)
            ).pow(2)
            tensor[..., -fade_samples:] *= fade_curve

        # Save as WAV first, then convert if needed
        wav_path = Path(tempfile.mktemp(suffix=".wav"))
        torchaudio.save(str(wav_path), tensor, sr)

        if output_format == "wav":
            return CogPath(wav_path)

        import subprocess

        out_path = Path(tempfile.mktemp(suffix=f".{output_format}"))
        subprocess.check_call(
            ["ffmpeg", "-y", "-i", str(wav_path), "-q:a", "2", str(out_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wav_path.unlink(missing_ok=True)
        return CogPath(out_path)
