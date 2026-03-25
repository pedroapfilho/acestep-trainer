"""CLI entry point for acestep-trainer."""

import typer

app = typer.Typer(name="acestep-train", help="ACE-Step remote training pipeline")


@app.command()
def status(
    bucket: str = typer.Argument(help="HF bucket name (user/repo)"),
):
    """Show dataset state summary from the bucket."""
    from acestep_trainer.state import load_state

    state = load_state(bucket)
    total = len(state.samples)
    unlabeled = len(state.get_by_status("unlabeled"))
    labeled = len(state.get_by_status("labeled"))
    preprocessed = len(state.get_by_status("preprocessed"))

    typer.echo(f"Dataset: {state.name}")
    typer.echo(f"Custom tag: {state.custom_tag} ({state.tag_position})")
    typer.echo(f"Total samples: {total}")
    typer.echo(f"  Unlabeled:    {unlabeled}")
    typer.echo(f"  Labeled:      {labeled}")
    typer.echo(f"  Preprocessed: {preprocessed}")


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
