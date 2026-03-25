"""Dataset state manager — tracks labeling/preprocessing/training progress.

The dataset.json file lives in the HF bucket and is the single source of truth.
Each sample has a 'status' field: unlabeled -> labeled -> preprocessed.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from acestep_trainer.bucket import file_exists, read_json, write_json

DATASET_JSON = "dataset.json"


@dataclass
class SampleState:
    """State for a single audio sample in the pipeline."""

    file: str = ""
    caption: str = ""
    genre: str = ""
    lyrics: str = ""
    bpm: int | None = None
    keyscale: str = ""
    timesignature: str = ""
    language: str = "unknown"
    is_instrumental: bool = True
    duration: float = 0.0
    status: str = "unlabeled"
    tensor_file: str = ""
    labeled_at: str = ""
    preprocessed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SampleState":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class DatasetState:
    """Full dataset state with metadata and samples."""

    name: str = "lofi-tracks"
    custom_tag: str = "lofi"
    tag_position: str = "prepend"
    num_samples: int = 0
    all_instrumental: bool = True
    genre_ratio: int = 50
    samples: list[SampleState] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": {
                "name": self.name,
                "custom_tag": self.custom_tag,
                "tag_position": self.tag_position,
                "num_samples": len(self.samples),
                "all_instrumental": self.all_instrumental,
                "genre_ratio": self.genre_ratio,
            },
            "samples": [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetState":
        meta = data.get("metadata", {})
        samples = [SampleState.from_dict(s) for s in data.get("samples", [])]
        return cls(
            name=meta.get("name", "lofi-tracks"),
            custom_tag=meta.get("custom_tag", "lofi"),
            tag_position=meta.get("tag_position", "prepend"),
            num_samples=len(samples),
            all_instrumental=meta.get("all_instrumental", True),
            genre_ratio=meta.get("genre_ratio", 50),
            samples=samples,
        )

    def get_by_status(self, status: str) -> list[SampleState]:
        return [s for s in self.samples if s.status == status]

    def get_file_set(self) -> set[str]:
        return {s.file for s in self.samples}

    def mark_labeled(self, file: str, **kwargs: Any) -> None:
        """Mark a sample as labeled with metadata."""
        for s in self.samples:
            if s.file == file:
                s.status = "labeled"
                s.labeled_at = datetime.now(timezone.utc).isoformat()
                for k, v in kwargs.items():
                    if hasattr(s, k):
                        setattr(s, k, v)
                return

    def mark_preprocessed(self, file: str, tensor_file: str) -> None:
        """Mark a sample as preprocessed with tensor path."""
        for s in self.samples:
            if s.file == file:
                s.status = "preprocessed"
                s.tensor_file = tensor_file
                s.preprocessed_at = datetime.now(timezone.utc).isoformat()
                return


def load_state(bucket: str) -> DatasetState:
    """Load dataset state from bucket, or create empty state."""
    if file_exists(bucket, DATASET_JSON):
        data = read_json(bucket, DATASET_JSON)
        state = DatasetState.from_dict(data)
        logger.info(
            f"Loaded state: {len(state.samples)} samples "
            f"({len(state.get_by_status('labeled'))} labeled, "
            f"{len(state.get_by_status('preprocessed'))} preprocessed)"
        )
        return state

    logger.info("No dataset.json found, creating empty state")
    return DatasetState()


def save_state(bucket: str, state: DatasetState) -> None:
    """Save dataset state back to the bucket."""
    state.num_samples = len(state.samples)
    write_json(bucket, DATASET_JSON, state.to_dict())
    logger.info(f"Saved state: {len(state.samples)} samples")


def sync_files_to_state(bucket: str, state: DatasetState, audio_files: list[str]) -> int:
    """Add newly discovered audio files to state. Returns count of new files."""
    existing = state.get_file_set()
    new_count = 0

    for f in audio_files:
        if f not in existing:
            state.samples.append(SampleState(file=f))
            new_count += 1

    if new_count > 0:
        logger.info(f"Added {new_count} new files to state")

    return new_count
