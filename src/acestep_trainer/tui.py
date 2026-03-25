"""Interactive TUI for ACE-Step training pipeline."""

from __future__ import annotations

import subprocess
from typing import Any

from textual import on
from textual import work
from textual.app import App
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button
from textual.widgets import Checkbox
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import Select
from textual.widgets import Static
from textual.widgets import TabbedContent
from textual.widgets import TabPane

GPU_FLAVORS = [
    ("a10g-small — 24GB $1.00/hr", "a10g-small"),
    ("a10g-large — 24GB $1.50/hr", "a10g-large"),
    ("l40s — 48GB $1.80/hr", "l40s"),
    ("a100-large — 80GB $2.50/hr", "a100-large"),
    ("h200 — 141GB $5.00/hr", "h200"),
]


class InitBucketModal(ModalScreen[dict[str, Any] | None]):
    """Modal for initializing a new bucket."""

    CSS = """
    InitBucketModal {
        align: center middle;
    }
    #init-dialog {
        width: 70;
        height: auto;
        max-height: 30;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #init-dialog Label {
        margin-top: 1;
    }
    #init-dialog Input {
        margin-bottom: 0;
    }
    #init-buttons {
        margin-top: 1;
        align: right middle;
        height: 3;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="init-dialog"):
            yield Label("Initialize New Bucket", classes="title")
            yield Label("Bucket ID (user/name):")
            yield Input(placeholder="pedroapfilho/my-dataset", id="bucket-id")
            yield Label("Dataset name:")
            yield Input(placeholder="My Lo-fi Dataset", id="dataset-name")
            yield Label("Custom tag (prepended to captions):")
            yield Input(placeholder="lofi", id="custom-tag")
            yield Checkbox("All instrumental (no vocals)", value=True, id="all-instrumental")
            yield Label("Genre ratio (0-100, % using genre vs caption):")
            yield Input(value="50", id="genre-ratio")
            with Horizontal(id="init-buttons"):
                yield Button("Create", variant="primary", id="btn-create")
                yield Button("Cancel", id="btn-cancel")

    @on(Button.Pressed, "#btn-create")
    def handle_create(self) -> None:
        bucket_id = self.query_one("#bucket-id", Input).value.strip()
        name = self.query_one("#dataset-name", Input).value.strip()
        tag = self.query_one("#custom-tag", Input).value.strip()
        instrumental = self.query_one("#all-instrumental", Checkbox).value
        ratio_str = self.query_one("#genre-ratio", Input).value.strip()
        ratio = int(ratio_str) if ratio_str.isdigit() else 50

        if not bucket_id or not name:
            self.notify("Bucket ID and name are required", severity="error")
            return

        self.dismiss(
            {
                "bucket_id": bucket_id,
                "name": name,
                "custom_tag": tag,
                "all_instrumental": instrumental,
                "genre_ratio": ratio,
            }
        )

    @on(Button.Pressed, "#btn-cancel")
    def handle_cancel(self) -> None:
        self.dismiss(None)


class SubmitJobModal(ModalScreen[dict[str, Any] | None]):
    """Modal for configuring and submitting a job."""

    CSS = """
    SubmitJobModal {
        align: center middle;
    }
    #submit-dialog {
        width: 70;
        height: auto;
        max-height: 35;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #submit-dialog Label {
        margin-top: 1;
    }
    #submit-buttons {
        margin-top: 1;
        align: right middle;
        height: 3;
    }
    """

    def __init__(self, bucket: str, phase: str = "label") -> None:
        super().__init__()
        self._bucket = bucket
        self._phase = phase

    def compose(self) -> ComposeResult:
        with Vertical(id="submit-dialog"):
            yield Label(f"Submit {self._phase.title()} Job", classes="title")
            yield Label("Bucket:")
            yield Input(value=self._bucket, id="job-bucket", disabled=True)
            yield Label("GPU flavor:")
            yield Select(GPU_FLAVORS, value="a10g-large", id="job-flavor")

            if self._phase == "label":
                yield Label("Parallel shards:")
                yield Input(value="8", id="job-parallel")
                yield Label("Max samples (0 = all):")
                yield Input(value="0", id="job-max-samples")
            elif self._phase == "preprocess":
                yield Label("Max samples (0 = all):")
                yield Input(value="0", id="job-max-samples")
            elif self._phase == "train":
                yield Label("Output model repo (user/name):")
                yield Input(placeholder="pedroapfilho/acestep-lofi-lora", id="job-output-repo")
                yield Label("LoRA rank:")
                yield Input(value="8", id="job-lora-rank")
                yield Label("Max epochs:")
                yield Input(value="100", id="job-max-epochs")

            yield Label("Timeout:")
            default_timeout = {"label": "12h", "preprocess": "24h", "train": "12h"}
            yield Input(value=default_timeout.get(self._phase, "12h"), id="job-timeout")

            with Horizontal(id="submit-buttons"):
                yield Button("Submit", variant="primary", id="btn-submit")
                yield Button("Dry Run", variant="warning", id="btn-dry-run")
                yield Button("Cancel", id="btn-cancel-job")

    def _build_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "bucket": self._bucket,
            "phase": self._phase,
            "flavor": str(self.query_one("#job-flavor", Select).value),
            "timeout": self.query_one("#job-timeout", Input).value.strip(),
        }
        if self._phase == "label":
            config["parallel"] = int(self.query_one("#job-parallel", Input).value.strip() or "1")
            config["max_samples"] = int(
                self.query_one("#job-max-samples", Input).value.strip() or "0"
            )
        elif self._phase == "preprocess":
            config["max_samples"] = int(
                self.query_one("#job-max-samples", Input).value.strip() or "0"
            )
        elif self._phase == "train":
            config["output_repo"] = self.query_one("#job-output-repo", Input).value.strip()
            config["lora_rank"] = int(self.query_one("#job-lora-rank", Input).value.strip() or "8")
            config["max_epochs"] = int(
                self.query_one("#job-max-epochs", Input).value.strip() or "100"
            )
        return config

    @on(Button.Pressed, "#btn-submit")
    def handle_submit(self) -> None:
        config = self._build_config()
        config["dry_run"] = False
        self.dismiss(config)

    @on(Button.Pressed, "#btn-dry-run")
    def handle_dry_run(self) -> None:
        config = self._build_config()
        config["dry_run"] = True
        self.dismiss(config)

    @on(Button.Pressed, "#btn-cancel-job")
    def handle_cancel(self) -> None:
        self.dismiss(None)


class AceStepTUI(App[None]):
    """ACE-Step Training Pipeline TUI."""

    CSS = """
    Screen {
        background: $surface;
    }
    #status-panel {
        height: auto;
        min-height: 8;
        border: solid $primary;
        margin: 1;
        padding: 1;
    }
    #actions-bar {
        height: 3;
        margin: 0 1;
    }
    #actions-bar Button {
        margin-right: 1;
    }
    #jobs-table {
        height: 1fr;
        margin: 1;
    }
    .stat-label {
        color: $text-muted;
    }
    .stat-value {
        color: $text;
        text-style: bold;
    }
    .progress-bar {
        margin-top: 1;
    }
    #log-panel {
        height: 1fr;
        margin: 1;
        border: solid $primary;
        padding: 1;
        overflow-y: auto;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("i", "init_bucket", "Init Bucket"),
        Binding("l", "submit_label", "Label"),
        Binding("p", "submit_preprocess", "Preprocess"),
        Binding("t", "submit_train", "Train"),
        Binding("m", "merge_shards", "Merge"),
    ]

    TITLE = "ACE-Step Trainer"

    def __init__(self, bucket: str = "") -> None:
        super().__init__()
        self._bucket = bucket
        self._status_data: dict[str, int] = {}
        self._log_lines: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Dashboard", id="tab-dashboard"):
                yield Static(id="status-panel")
                with Horizontal(id="actions-bar"):
                    yield Button("Refresh", variant="default", id="btn-refresh")
                    yield Button("Init Bucket", variant="primary", id="btn-init")
                    yield Button("Scan", variant="default", id="btn-scan")
                    yield Button("Label", variant="success", id="btn-label")
                    yield Button("Preprocess", variant="warning", id="btn-preprocess")
                    yield Button("Train", variant="error", id="btn-train")
                    yield Button("Merge", variant="default", id="btn-merge")
                with Container(id="jobs-table"):
                    yield DataTable()
            with TabPane("Logs", id="tab-logs"):
                yield Static(id="log-panel")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Status", "ID", "Phase", "Duration", "Message")
        if self._bucket:
            self.refresh_data()

    @work(thread=True)
    def refresh_data(self) -> None:
        """Fetch dataset status and jobs."""
        if not self._bucket:
            self.call_from_thread(self._update_status_no_bucket)
            return

        # Fetch dataset status
        try:
            from acestep_trainer.state import load_state

            state = load_state(self._bucket)
            total = len(state.samples)
            unlabeled = len(state.get_by_status("unlabeled"))
            labeled = len(state.get_by_status("labeled"))
            preprocessed = len(state.get_by_status("preprocessed"))

            self._status_data = {
                "total": total,
                "unlabeled": unlabeled,
                "labeled": labeled,
                "preprocessed": preprocessed,
            }
            self.call_from_thread(self._update_status_display)
        except Exception as e:
            self.call_from_thread(self._update_status_error, str(e))

        # Fetch jobs
        try:
            from datetime import datetime
            from datetime import timezone

            from huggingface_hub import list_jobs

            jobs = list(list_jobs())
            rows: list[tuple[str, ...]] = []
            for job in jobs:
                cmd = " ".join(job.command) if job.command else ""
                if "acestep" not in cmd:
                    continue
                stage = str(job.status.stage).split(".")[-1]
                icon = {"RUNNING": "🟢", "COMPLETED": "✅", "ERROR": "❌", "CANCELED": "⚪"}.get(
                    stage, "?"
                )
                short_id = job.id[:12]
                # Extract phase
                phase = "unknown"
                if "--shard-id" in cmd:
                    parts = cmd.split()
                    for i, p in enumerate(parts):
                        if p == "--shard-id" and i + 1 < len(parts):
                            sid = parts[i + 1]
                            ns = "?"
                            for j, q in enumerate(parts):
                                if q == "--num-shards" and j + 1 < len(parts):
                                    ns = parts[j + 1]
                            phase = f"shard {sid}/{ns}"
                elif "label.py" in cmd:
                    phase = "label"
                elif "preprocess.py" in cmd:
                    phase = "preprocess"
                elif "train.py" in cmd:
                    phase = "train"

                now = datetime.now(timezone.utc)
                created = job.created_at or now
                delta = now - created
                minutes = int(delta.total_seconds() // 60)
                if minutes >= 60:
                    duration = f"{minutes // 60}h{minutes % 60:02d}m"
                else:
                    duration = f"{minutes}m"

                msg = (job.status.message or "")[:40]
                rows.append((f"{icon} {stage}", short_id, phase, duration, msg))

            self.call_from_thread(self._update_jobs_table, rows)
        except Exception:
            pass

    def _update_status_no_bucket(self) -> None:
        panel = self.query_one("#status-panel", Static)
        panel.update(
            "No bucket configured.\n\n"
            "Press [b]i[/b] to initialize a new bucket, or start with:\n"
            "  uv run acestep-train tui <bucket-id>"
        )

    def _update_status_display(self) -> None:
        d = self._status_data
        total = d.get("total", 0)
        unlabeled = d.get("unlabeled", 0)
        labeled = d.get("labeled", 0)
        preprocessed = d.get("preprocessed", 0)

        pct = (labeled + preprocessed) / total * 100 if total > 0 else 0
        bar_len = 40
        filled = int(bar_len * (labeled + preprocessed) / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)

        panel = self.query_one("#status-panel", Static)
        panel.update(
            f"[bold]Bucket:[/bold] {self._bucket}\n"
            f"[bold]Total:[/bold] {total:,}    "
            f"[dim]Unlabeled:[/dim] {unlabeled:,}    "
            f"[green]Labeled:[/green] {labeled:,}    "
            f"[yellow]Preprocessed:[/yellow] {preprocessed:,}\n\n"
            f"[{bar}] {pct:.1f}%"
        )

    def _update_status_error(self, error: str) -> None:
        panel = self.query_one("#status-panel", Static)
        panel.update(f"[red]Error loading status:[/red] {error}")

    def _update_jobs_table(self, rows: list[tuple[str, ...]]) -> None:
        table = self.query_one(DataTable)
        table.clear()
        for row in rows:
            table.add_row(*row)

    def _append_log(self, text: str) -> None:
        self._log_lines.append(text)
        panel = self.query_one("#log-panel", Static)
        panel.update("\n".join(self._log_lines))

    @on(Button.Pressed, "#btn-refresh")
    def handle_refresh(self) -> None:
        self.refresh_data()

    def action_refresh(self) -> None:
        self.refresh_data()

    @on(Button.Pressed, "#btn-init")
    def handle_init(self) -> None:
        self.action_init_bucket()

    def action_init_bucket(self) -> None:
        self.push_screen(InitBucketModal(), self._on_init_result)

    def _on_init_result(self, result: dict[str, Any] | None) -> None:
        if result is None:
            return
        self._do_init_bucket(result)

    @work(thread=True)
    def _do_init_bucket(self, config: dict[str, Any]) -> None:
        from acestep_trainer.bucket_init import bucket_exists
        from acestep_trainer.bucket_init import create_bucket
        from acestep_trainer.bucket_init import init_bucket

        bucket_id = config["bucket_id"]

        if not bucket_exists(bucket_id):
            self.call_from_thread(self.notify, f"Creating bucket {bucket_id}...")
            if not create_bucket(bucket_id):
                self.call_from_thread(self.notify, "Failed to create bucket", severity="error")
                return

        init_bucket(
            bucket_id,
            name=config["name"],
            custom_tag=config["custom_tag"],
            all_instrumental=config["all_instrumental"],
            genre_ratio=config["genre_ratio"],
        )
        self._bucket = bucket_id
        self.call_from_thread(self.notify, f"Bucket {bucket_id} initialized!")
        self.call_from_thread(self.refresh_data)

    @on(Button.Pressed, "#btn-scan")
    def handle_scan(self) -> None:
        if not self._bucket:
            self.notify("No bucket configured", severity="error")
            return
        self._do_scan()

    @work(thread=True)
    def _do_scan(self) -> None:
        from acestep_trainer.bucket import list_audio_files
        from acestep_trainer.state import load_state
        from acestep_trainer.state import save_state
        from acestep_trainer.state import sync_files_to_state

        self.call_from_thread(self.notify, "Scanning bucket...")
        state = load_state(self._bucket)
        audio_files = list_audio_files(self._bucket)
        new_count = sync_files_to_state(self._bucket, state, audio_files)
        if new_count > 0:
            save_state(self._bucket, state)
        self.call_from_thread(self.notify, f"Found {len(audio_files)} files, {new_count} new")
        self.call_from_thread(self.refresh_data)

    @on(Button.Pressed, "#btn-label")
    def handle_label(self) -> None:
        self.action_submit_label()

    @on(Button.Pressed, "#btn-preprocess")
    def handle_preprocess(self) -> None:
        self.action_submit_preprocess()

    @on(Button.Pressed, "#btn-train")
    def handle_train(self) -> None:
        self.action_submit_train()

    def action_submit_label(self) -> None:
        if not self._bucket:
            self.notify("No bucket configured", severity="error")
            return
        self.push_screen(SubmitJobModal(self._bucket, "label"), self._on_submit_result)

    def action_submit_preprocess(self) -> None:
        if not self._bucket:
            self.notify("No bucket configured", severity="error")
            return
        self.push_screen(SubmitJobModal(self._bucket, "preprocess"), self._on_submit_result)

    def action_submit_train(self) -> None:
        if not self._bucket:
            self.notify("No bucket configured", severity="error")
            return
        self.push_screen(SubmitJobModal(self._bucket, "train"), self._on_submit_result)

    def _on_submit_result(self, result: dict[str, Any] | None) -> None:
        if result is None:
            return
        self._do_submit_job(result)

    @work(thread=True)
    def _do_submit_job(self, config: dict[str, Any]) -> None:
        """Submit job(s) via the submit_job script."""
        phase = config["phase"]
        bucket = config["bucket"]
        flavor = config["flavor"]
        timeout = config["timeout"]
        dry_run = config.get("dry_run", False)

        cmd = [
            "python",
            "scripts/submit_job.py",
            phase,
            "--bucket",
            bucket,
            "--flavor",
            flavor,
            "--timeout",
            timeout,
        ]

        if phase == "label":
            parallel = config.get("parallel", 1)
            max_samples = config.get("max_samples", 0)
            if parallel > 1:
                cmd.extend(["--parallel", str(parallel)])
            if max_samples > 0:
                cmd.extend(["--max-samples", str(max_samples)])
        elif phase == "preprocess":
            max_samples = config.get("max_samples", 0)
            if max_samples > 0:
                cmd.extend(["--max-samples", str(max_samples)])
        elif phase == "train":
            cmd.extend(["--output-repo", config.get("output_repo", "")])
            cmd.extend(["--lora-rank", str(config.get("lora_rank", 8))])
            cmd.extend(["--max-epochs", str(config.get("max_epochs", 100))])

        if dry_run:
            cmd.append("--dry-run")

        self.call_from_thread(self._append_log, f"$ {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        self.call_from_thread(self._append_log, result.stdout)
        if result.stderr:
            self.call_from_thread(self._append_log, result.stderr)

        if result.returncode == 0:
            label = "DRY RUN" if dry_run else "submitted"
            self.call_from_thread(self.notify, f"Job {label} successfully!")
        else:
            self.call_from_thread(self.notify, "Job submission failed", severity="error")

        self.call_from_thread(self.refresh_data)

    @on(Button.Pressed, "#btn-merge")
    def handle_merge(self) -> None:
        self.action_merge_shards()

    def action_merge_shards(self) -> None:
        if not self._bucket:
            self.notify("No bucket configured", severity="error")
            return
        self._do_merge()

    @work(thread=True)
    def _do_merge(self) -> None:
        from acestep_trainer.bucket import file_exists
        from acestep_trainer.bucket import read_json
        from acestep_trainer.state import SampleState
        from acestep_trainer.state import load_state
        from acestep_trainer.state import save_state

        self.call_from_thread(self.notify, "Merging shards...")

        state = load_state(self._bucket)
        existing_files = {s.file: s for s in state.samples}
        total_merged = 0

        # Auto-detect shards
        shard_id = 0
        while True:
            shard_path = f"labels_shard_{shard_id}.json"
            if not file_exists(self._bucket, shard_path):
                break

            shard_data = read_json(self._bucket, shard_path)
            shard_samples = [SampleState.from_dict(s) for s in shard_data.get("samples", [])]
            applied = 0
            for ls in shard_samples:
                if ls.status != "labeled":
                    continue
                target = existing_files.get(ls.file)
                if target and target.status == "unlabeled":
                    target.status = ls.status
                    target.caption = ls.caption
                    target.genre = ls.genre
                    target.lyrics = ls.lyrics
                    target.bpm = ls.bpm
                    target.keyscale = ls.keyscale
                    target.timesignature = ls.timesignature
                    target.language = ls.language
                    target.is_instrumental = ls.is_instrumental
                    target.labeled_at = ls.labeled_at
                    applied += 1

            self.call_from_thread(self._append_log, f"Shard {shard_id}: {applied} labels applied")
            total_merged += applied
            shard_id += 1

        if total_merged > 0:
            save_state(self._bucket, state)

        self.call_from_thread(
            self.notify,
            f"Merged {total_merged} labels from {shard_id} shards",
        )
        self.call_from_thread(self.refresh_data)


def run_tui(bucket: str = "") -> None:
    """Launch the TUI app."""
    app = AceStepTUI(bucket=bucket)
    app.run()
