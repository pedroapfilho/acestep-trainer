#!/usr/bin/env python3
"""Patch torchaudio to use soundfile instead of torchcodec.

torchcodec in ace-step's deps requires CUDA 13 runtime (libnvrtc.so.13)
which isn't available in CUDA 12.8 Docker images. This replaces the
torchcodec backend with a soundfile-based implementation.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types


def _load_with_soundfile(
    uri: str,
    *_args: object,
    **_kwargs: object,
) -> tuple:
    """Load audio using soundfile + torch, matching torchaudio.load() return."""
    import numpy as np
    import soundfile as sf
    import torch

    data, sr = sf.read(uri, dtype="float32", always_2d=True)
    # soundfile returns (samples, channels), torch wants (channels, samples)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))
    return waveform, sr


def _save_with_soundfile(
    uri: str,
    src: object,
    sample_rate: int,
    *_args: object,
    **_kwargs: object,
) -> None:
    """Save audio using soundfile, matching torchaudio.save() interface."""
    import numpy as np
    import soundfile as sf
    import torch

    if isinstance(src, torch.Tensor):
        data = src.cpu().numpy().T  # (channels, samples) -> (samples, channels)
    else:
        data = np.array(src).T
    sf.write(uri, data, sample_rate)


def patch() -> None:
    """Replace torchcodec with soundfile-based audio I/O."""

    def _make_stub(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        mod.__loader__ = None  # type: ignore[assignment]
        mod.__path__ = []  # type: ignore[attr-defined]
        mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
        return mod

    # Replace torchaudio's torchcodec wrapper with soundfile implementations
    ta_stub = _make_stub("torchaudio._torchcodec")
    ta_stub.load_with_torchcodec = _load_with_soundfile  # type: ignore[attr-defined]
    ta_stub.save_with_torchcodec = _save_with_soundfile  # type: ignore[attr-defined]
    sys.modules["torchaudio._torchcodec"] = ta_stub

    # Stub torchcodec package to prevent import errors elsewhere
    for mod_name in [
        "torchcodec",
        "torchcodec.decoders",
        "torchcodec._internally_replaced_utils",
    ]:
        sys.modules[mod_name] = _make_stub(mod_name)

    print("Patched torchaudio: using soundfile backend instead of torchcodec")


if __name__ == "__main__":
    patch()
