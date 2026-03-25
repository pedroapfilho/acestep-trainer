#!/usr/bin/env python3
"""Patch torchaudio to disable torchcodec backend.

torchcodec in ace-step's deps requires CUDA 13 runtime (libnvrtc.so.13)
which isn't available in CUDA 12.8 Docker images. This patch disables
torchcodec so torchaudio falls back to the ffmpeg/soundfile backend.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types


def patch() -> None:
    """Stub torchcodec modules so torchaudio skips them."""

    def _make_stub(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        mod.__loader__ = None  # type: ignore[assignment]
        mod.__path__ = []  # type: ignore[attr-defined]
        mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
        return mod

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("torchcodec disabled — using ffmpeg backend")

    # Stub torchaudio's internal torchcodec wrapper
    ta_stub = _make_stub("torchaudio._torchcodec")
    ta_stub.load_with_torchcodec = _raise  # type: ignore[attr-defined]
    ta_stub.save_with_torchcodec = _raise  # type: ignore[attr-defined]
    sys.modules["torchaudio._torchcodec"] = ta_stub

    # Stub torchcodec package
    for mod_name in [
        "torchcodec",
        "torchcodec.decoders",
        "torchcodec._internally_replaced_utils",
    ]:
        sys.modules[mod_name] = _make_stub(mod_name)

    print("Patched torchaudio: torchcodec backend disabled, using ffmpeg/soundfile")


if __name__ == "__main__":
    patch()
