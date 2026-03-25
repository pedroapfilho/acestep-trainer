"""Cog Predictor for ACE-Step 1.5 with LoRA support.

Deploys to Replicate as a serverless music generation model.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torchaudio
from cog import BasePredictor, Input, Path as CogPath


class Predictor(BasePredictor):
    def setup(self):
        """Download ACE-Step model and LoRA weights."""
        from huggingface_hub import snapshot_download

        # Download base ACE-Step model
        self.checkpoints_dir = "/src/checkpoints"
        if not os.path.exists(os.path.join(self.checkpoints_dir, "acestep-v15-turbo")):
            snapshot_download(
                "ACE-Step/Ace-Step1.5",
                local_dir=self.checkpoints_dir,
            )

        # Download default LoRA
        self.lora_dir = "/src/lora"
        lora_repo = os.environ.get("LORA_REPO", "pedroapfilho/acestep-lofi-lora")
        lora_subfolder = os.environ.get("LORA_SUBFOLDER", "final/adapter")
        if not os.path.exists(os.path.join(self.lora_dir, "adapter_config.json")):
            snapshot_download(
                lora_repo,
                local_dir="/src/lora_full",
                allow_patterns=f"{lora_subfolder}/*",
            )
            # Move subfolder contents to lora_dir
            src = os.path.join("/src/lora_full", lora_subfolder)
            os.makedirs(self.lora_dir, exist_ok=True)
            for f in os.listdir(src):
                os.rename(os.path.join(src, f), os.path.join(self.lora_dir, f))

        # Sync model code files
        self._sync_model_code()

        # Load the model
        sys.path.insert(0, "/src/ace-step-1.5")
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
        lora_status = self.handler.load_lora(self.lora_dir)
        print(f"LoRA loaded: {lora_status}")

    def _sync_model_code(self):
        """Clone ace-step repo for model code if not present."""
        if not os.path.exists("/src/ace-step-1.5"):
            subprocess.run(
                ["git", "clone", "https://github.com/ace-step/ACE-Step-1.5.git", "/src/ace-step-1.5"],
                check=True,
            )
        # Symlink checkpoints into the repo
        repo_checkpoints = "/src/ace-step-1.5/checkpoints"
        if not os.path.exists(repo_checkpoints):
            os.symlink(self.checkpoints_dir, repo_checkpoints)

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
        )

        audios = result.get("audios", [])
        if not audios:
            raise RuntimeError("Generation failed — no audio output")

        audio_data = audios[0]
        tensor = audio_data["tensor"]
        sr = audio_data["sample_rate"]

        out_path = Path(tempfile.mktemp(suffix=".wav"))
        torchaudio.save(str(out_path), tensor, sr)
        return CogPath(out_path)
