"""CLI entry point for acestep-trainer."""

import time
from datetime import datetime

import typer

app = typer.Typer(name="acestep-train", help="ACE-Step remote training pipeline")

MERGE_INTERVAL = 300  # 5 minutes


def _run_merge(bucket: str) -> int:
    """Run merge and return count of labels applied."""
    from acestep_trainer.bucket import file_exists
    from acestep_trainer.bucket import read_json
    from acestep_trainer.state import SampleState
    from acestep_trainer.state import load_state
    from acestep_trainer.state import save_state

    state = load_state(bucket)
    existing_files = {s.file: s for s in state.samples}
    total_merged = 0
    shard_id = 0

    while True:
        shard_path = f"labels_shard_{shard_id}.json"
        if not file_exists(bucket, shard_path):
            break

        shard_data = read_json(bucket, shard_path)
        shard_samples = [SampleState.from_dict(s) for s in shard_data.get("samples", [])]

        for labeled_sample in shard_samples:
            if labeled_sample.status != "labeled":
                continue
            target = existing_files.get(labeled_sample.file)
            if target and target.status == "unlabeled":
                target.status = labeled_sample.status
                target.caption = labeled_sample.caption
                target.genre = labeled_sample.genre
                target.lyrics = labeled_sample.lyrics
                target.bpm = labeled_sample.bpm
                target.keyscale = labeled_sample.keyscale
                target.timesignature = labeled_sample.timesignature
                target.language = labeled_sample.language
                target.is_instrumental = labeled_sample.is_instrumental
                target.labeled_at = labeled_sample.labeled_at
                total_merged += 1

        shard_id += 1

    if total_merged > 0:
        save_state(bucket, state)

    return total_merged


def _print_status(bucket: str, *, clear: bool = False, merge_info: str = "") -> None:
    from acestep_trainer.state import load_state

    state = load_state(bucket)
    total = len(state.samples)
    unlabeled = len(state.get_by_status("unlabeled"))
    labeled = len(state.get_by_status("labeled"))
    preprocessed = len(state.get_by_status("preprocessed"))

    if clear:
        typer.echo("\033[2J\033[H", nl=False)

    typer.echo(f"Dataset: {state.name}")
    typer.echo(f"Custom tag: {state.custom_tag} ({state.tag_position})")
    typer.echo(f"Total samples: {total}")
    typer.echo(f"  Unlabeled:    {unlabeled}")
    typer.echo(f"  Labeled:      {labeled}")
    typer.echo(f"  Preprocessed: {preprocessed}")

    if total > 0:
        pct = (labeled + preprocessed) / total * 100
        bar_len = 30
        filled = int(bar_len * (labeled + preprocessed) / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        typer.echo(f"  Progress:     [{bar}] {pct:.1f}%")

    typer.echo(f"  Updated:      {datetime.now().strftime('%H:%M:%S')}")
    if merge_info:
        typer.echo(f"  Last merge:   {merge_info}")


@app.command()
def status(
    bucket: str = typer.Argument(help="HF bucket name (user/repo)"),
    live: bool = typer.Option(False, "--live", help="Poll every 30s, merge every 5min"),
    interval: int = typer.Option(30, "--interval", help="Poll interval in seconds (with --live)"),
):
    """Show dataset state summary from the bucket."""
    if not live:
        _print_status(bucket)
        return

    last_merge = 0.0
    merge_info = ""

    try:
        while True:
            now = time.time()
            if now - last_merge >= MERGE_INTERVAL:
                merged = _run_merge(bucket)
                last_merge = now
                if merged > 0:
                    merge_info = f"+{merged} labels at {datetime.now().strftime('%H:%M:%S')}"
                else:
                    merge_info = f"no new labels at {datetime.now().strftime('%H:%M:%S')}"

            _print_status(bucket, clear=True, merge_info=merge_info)
            time.sleep(interval)
    except KeyboardInterrupt:
        typer.echo("\nStopped.")


@app.command()
def scan(
    bucket: str = typer.Argument(help="HF bucket name (user/repo)"),
):
    """Scan bucket for audio files and sync to dataset.json."""
    from acestep_trainer.bucket import list_audio_files
    from acestep_trainer.state import load_state
    from acestep_trainer.state import save_state
    from acestep_trainer.state import sync_files_to_state

    state = load_state(bucket)

    typer.echo("Scanning bucket for audio files...")
    audio_files = list_audio_files(bucket)
    typer.echo(f"Found {len(audio_files)} audio files")

    new_count = sync_files_to_state(bucket, state, audio_files)
    if new_count > 0:
        save_state(bucket, state)
        typer.echo(f"Added {new_count} new files to dataset.json")
    else:
        typer.echo("No new files found")


@app.command()
def merge(
    bucket: str = typer.Argument(help="HF bucket name (user/repo)"),
):
    """Merge label shards into dataset.json after parallel labeling.

    Auto-detects all labels_shard_*.json files in the bucket.
    """
    merged = _run_merge(bucket)

    if merged > 0:
        typer.echo(f"Merged {merged} labels")
    else:
        typer.echo("No new labels to merge")

    _print_status(bucket)


if __name__ == "__main__":
    app()
