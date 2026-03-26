"""Cog Predictor for ACE-Step 1.5 with LoRA support.

Deploys to Replicate as a serverless music generation model.
Model weights and LoRA are baked into the Docker image at build time.
"""

import os
import tempfile
from pathlib import Path

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
