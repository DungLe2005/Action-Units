"""
utils/rich_callback.py
-----------------------
Lightning Callback hiển thị training progress đẹp với Rich:
  • Mỗi train step: thanh progress + loss hiện tại
  • Mỗi val epoch : bảng F1/AUC per-AU + summary tổng
  • Màu sắc tự động theo giá trị metric (xanh/vàng/đỏ)
"""

from __future__ import annotations

import time
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import Callback
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich import box

# Dùng chung console với rich_logger nếu đã import, không thì tạo mới
try:
    from utils.rich_logger import console
except ImportError:
    console = Console(highlight=True)

AU_IDS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]


def _color_f1(val: float) -> str:
    """Màu cho F1 score."""
    if val >= 0.70:
        return f"[bold green]{val:.4f}[/bold green]"
    elif val >= 0.50:
        return f"[yellow]{val:.4f}[/yellow]"
    else:
        return f"[red]{val:.4f}[/red]"


def _color_auc(val: float) -> str:
    """Màu cho AUC score."""
    if val >= 0.80:
        return f"[bold green]{val:.4f}[/bold green]"
    elif val >= 0.65:
        return f"[yellow]{val:.4f}[/yellow]"
    else:
        return f"[red]{val:.4f}[/red]"


class RichTrainingCallback(Callback):
    """
    Callback hiển thị log đẹp trong quá trình training.
    Thêm vào Trainer:
        trainer = L.Trainer(callbacks=[RichTrainingCallback(), ...])
    """

    def __init__(self):
        super().__init__()
        self._train_start_time: float = 0.0
        self._epoch_start_time: float = 0.0
        self._progress: Progress | None = None
        self._train_task = None
        self._loss_history: list[float] = []

    # ── Setup progress bar ────────────────────────────────────────────────

    def _make_progress(self) -> Progress:
        return Progress(
            SpinnerColumn(style="bold cyan"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=36, style="cyan", complete_style="bold green"),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("• loss [bold magenta]{task.fields[loss]:.4f}"),
            console=console,
            refresh_per_second=5,
        )

    # ── Training lifecycle ────────────────────────────────────────────────

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self._train_start_time = time.time()
        total_epochs = trainer.max_epochs
        num_gpus = trainer.num_devices
        precision = trainer.precision

        console.print(
            Panel(
                f"[bold cyan]AU Detection Training[/bold cyan]\n"
                f"Epochs   : [yellow]{total_epochs}[/yellow]\n"
                f"GPUs     : [yellow]{num_gpus}[/yellow]\n"
                f"Precision: [yellow]{precision}[/yellow]\n"
                f"Strategy : [yellow]{type(trainer.strategy).__name__}[/yellow]",
                title="🚀 Training Started",
                border_style="bold blue",
                expand=False,
            )
        )

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self._epoch_start_time = time.time()
        self._loss_history.clear()
        epoch = trainer.current_epoch + 1
        total = trainer.max_epochs
        num_steps = trainer.num_training_batches

        self._progress = self._make_progress()
        self._progress.start()
        self._train_task = self._progress.add_task(
            f"Epoch {epoch}/{total}",
            total=num_steps,
            loss=0.0,
        )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        if self._progress is None or self._train_task is None:
            return

        loss_val = float(trainer.callback_metrics.get("train/loss_step", 0.0))
        self._loss_history.append(loss_val)

        self._progress.update(self._train_task, advance=1, loss=loss_val)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if self._progress:
            self._progress.stop()
            self._progress = None

        elapsed = time.time() - self._epoch_start_time
        epoch = trainer.current_epoch + 1
        avg_loss = (
            sum(self._loss_history) / len(self._loss_history)
            if self._loss_history
            else 0.0
        )
        lr = trainer.callback_metrics.get(
            "lr-Adam", trainer.optimizers[0].param_groups[0]["lr"]
        )

        console.print(
            f"  [bold white on blue] Epoch {epoch:>3} [/bold white on blue] "
            f"avg_loss=[bold magenta]{avg_loss:.4f}[/bold magenta]  "
            f"lr=[cyan]{float(lr):.2e}[/cyan]  "
            f"time=[dim]{elapsed:.1f}s[/dim]"
        )

    # ── Validation ────────────────────────────────────────────────────────

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        metrics = trainer.callback_metrics

        avg_f1  = float(metrics.get("val/avg_f1",  0.0))
        avg_auc = float(metrics.get("val/avg_auc", 0.0))
        acc     = float(metrics.get("val/accuracy", 0.0))

        # ── Per-AU table ──────────────────────────────────────────────────
        table = Table(
            title=f"Validation – Epoch {trainer.current_epoch + 1}",
            box=box.ROUNDED,
            border_style="blue",
            show_header=True,
            header_style="bold white",
            padding=(0, 1),
        )
        table.add_column("AU",  style="bold cyan", justify="center", width=5)
        table.add_column("F1",  justify="center", width=10)
        table.add_column("AUC", justify="center", width=10)

        # Lấy per-AU metrics từ pl_module nếu có
        au_f1_list  = getattr(pl_module, "_last_au_f1",  [None] * 12)
        au_auc_list = getattr(pl_module, "_last_au_auc", [None] * 12)

        for i, au_id in enumerate(AU_IDS):
            f1_val  = au_f1_list[i]
            auc_val = au_auc_list[i]
            f1_str  = _color_f1(f1_val)   if f1_val  is not None else "[dim]N/A[/dim]"
            auc_str = _color_auc(auc_val) if auc_val is not None else "[dim]N/A[/dim]"
            table.add_row(f"AU{au_id}", f1_str, auc_str)

        # ── Summary row ───────────────────────────────────────────────────
        table.add_section()
        table.add_row(
            "[bold]AVG[/bold]",
            _color_f1(avg_f1),
            _color_auc(avg_auc),
        )

        console.print(table)
        console.print(
            f"  Accuracy: {_color_f1(acc)}  "
            f"Avg F1: {_color_f1(avg_f1)}  "
            f"Avg AUC: {_color_auc(avg_auc)}"
        )

    # ── Training complete ─────────────────────────────────────────────────

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        total_time = time.time() - self._train_start_time
        hours, rem = divmod(int(total_time), 3600)
        mins, secs = divmod(rem, 60)
        console.print(
            Panel(
                f"[bold green]Training Complete ✓[/bold green]\n"
                f"Total time: [yellow]{hours:02d}h {mins:02d}m {secs:02d}s[/yellow]",
                border_style="bold green",
                expand=False,
            )
        )
