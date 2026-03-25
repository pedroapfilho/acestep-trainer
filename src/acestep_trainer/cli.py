"""CLI entry point for acestep-trainer."""

import time
from datetime import datetime

import typer

app = typer.Typer(name="acestep-train", help="ACE-Step remote training pipeline")


def _print_status(bucket: str, *, clear: bool = False) -> None:
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


@app.command()
def status(
    bucket: str = typer.Argument(help="HF bucket name (user/repo)"),
    live: bool = typer.Option(False, "--live", help="Poll every 30s until interrupted"),
    interval: int = typer.Option(30, "--interval", help="Poll interval in seconds (with --live)"),
):
    """Show dataset state summary from the bucket."""
    if not live:
        _print_status(bucket)
        return

    try:
        while True:
            _print_status(bucket, clear=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        typer.echo("\nStopped.")


@app.command()
def scan(
    bucket: str = typer.Argument(help="HF bucket name (user/repo)"),
):
    """Scan bucket for audio files and sync to dataset.json."""
    from acestep_trainer.bucket import list_audio_files
    from acestep_trainer.state import load_state, save_state, sync_files_to_state

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


if __name__ == "__main__":
    app()
