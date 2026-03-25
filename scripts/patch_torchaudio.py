#!/usr/bin/env python3
"""Patch torchaudio to disable torchcodec backend.

torchcodec in ace-step's deps requires CUDA 13 runtime (libnvrtc.so.13)
which isn't available in CUDA 12.8 Docker images. This patch disables
torchcodec so torchaudio falls back to the ffmpeg/soundfile backend.
"""

from __future__ import annotations

import sys


def patch() -> None:
    # Stub out torchaudio's torchcodec integration
    import types

    stub = types.ModuleType("torchaudio._torchcodec")

    def load_with_torchcodec(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("torchcodec disabled — using ffmpeg backend")

    stub.load_with_torchcodec = load_with_torchcodec  # type: ignore[attr-defined]
    sys.modules["torchaudio._torchcodec"] = stub

    # Also stub torchcodec itself
    stub2 = types.ModuleType("torchcodec")
    sys.modules["torchcodec"] = stub2

    stub3 = types.ModuleType("torchcodec.decoders")
    sys.modules["torchcodec.decoders"] = stub3

    print("Patched torchaudio: torchcodec backend disabled, using ffmpeg/soundfile")


if __name__ == "__main__":
    patch()
