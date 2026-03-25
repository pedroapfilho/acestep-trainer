"""ACE-Step handler initialization for remote jobs.

Provides functions to initialize the DiT handler and LLM handler
for use in labeling, preprocessing, and training scripts.
The handler downloads models from HuggingFace on first use.
"""

import os
import sys

from loguru import logger

# ace-step-1.5 must be on sys.path — set by each script
# before importing this module.


def get_project_root() -> str:
    """Get the ace-step-1.5 project root directory."""
    # Check common locations
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "ace-step-1.5"),
        "/workspace/ace-step-1.5",
        os.environ.get("ACESTEP_ROOT", ""),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.abspath(c)

    raise FileNotFoundError(
        "ace-step-1.5 not found. Set ACESTEP_ROOT env var or ensure "
        "the submodule is checked out."
    )


def ensure_sys_path() -> str:
    """Add ace-step-1.5 to sys.path and return the project root."""
    root = get_project_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def init_dit_handler(
    device: str = "auto",
    quantization: str | None = None,
    offload_to_cpu: bool = False,
):
    """Initialize the ACE-Step DiT handler with all sub-models.

    For training, quantization must be None (LoRA is incompatible with
    quantized models).
    """
    root = ensure_sys_path()

    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    config_path = "acestep-v15-turbo"

    logger.info(f"Initializing DiT: config={config_path}, device={device}")
    status, ok = handler.initialize_service(
        project_root=root,
        config_path=config_path,
        device=device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=offload_to_cpu,
        offload_dit_to_cpu=False,
        quantization=quantization,
        prefer_source="huggingface",
    )

    if not ok:
        raise RuntimeError(f"Failed to initialize DiT handler: {status}")

    logger.info(f"DiT initialized: {status}")
    return handler


def init_llm_handler(
    device: str = "auto",
    model_name: str = "acestep-5Hz-lm-0.6B",
    backend: str = "pt",
):
    """Initialize the LLM handler for audio labeling/captioning.

    The LLM needs a checkpoint_dir (parent of model weights) and the
    model name (subdirectory). Downloads the model from HF if not present.
    """
    root = ensure_sys_path()
    checkpoint_dir = os.path.join(root, "checkpoints")

    # Download LM model if not present (DiT init only downloads the main model)
    from acestep.model_downloader import ensure_lm_model
    from pathlib import Path

    ok, msg = ensure_lm_model(
        model_name=model_name,
        checkpoints_dir=Path(checkpoint_dir),
        prefer_source="huggingface",
    )
    if not ok:
        raise RuntimeError(f"Failed to download LLM model: {msg}")
    logger.info(f"LLM model ready: {msg}")

    from acestep.llm_inference import LLMHandler

    logger.info(f"Initializing LLM: model={model_name}, device={device}, backend={backend}")

    llm = LLMHandler()
    status, ok = llm.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path=model_name,
        backend=backend,
        device=device,
    )

    if not ok:
        raise RuntimeError(f"Failed to initialize LLM handler: {status}")

    logger.info(f"LLM initialized: {status}")
    return llm
