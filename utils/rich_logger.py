"""
utils/rich_logger.py
---------------------
Logger dùng Rich cho output đẹp trên console.
Vẫn ghi file log đơn giản như logger.py gốc.

Sử dụng:
    from utils.rich_logger import setup_rich_logger, rprint, console
    logger = setup_rich_logger("transreid", output_dir, if_train=True)
"""

import logging
import os
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ── Global console instance (dùng chung toàn project) ───────────────────────
_THEME = Theme(
    {
        "info":    "bold cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error":   "bold red",
        "metric":  "bold magenta",
        "epoch":   "bold white on blue",
        "au":      "dim cyan",
    }
)

console = Console(theme=_THEME, highlight=True)


def rprint(*args, **kwargs):
    """Shortcut cho console.print() với theme."""
    console.print(*args, **kwargs)


# ── Setup logger ─────────────────────────────────────────────────────────────

def setup_rich_logger(
    name: str,
    save_dir: Optional[str] = None,
    if_train: bool = True,
) -> logging.Logger:
    """
    Tạo logger với:
      • Console handler   → Rich (màu sắc, icon, highlight tự động)
      • File handler      → plain text (giống logger.py gốc)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Tránh thêm handler trùng nếu gọi lại
    if logger.handlers:
        return logger

    # ── Rich console handler ──────────────────────────────────────────────
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
        log_time_format="[%H:%M:%S]",
    )
    rich_handler.setLevel(logging.DEBUG)
    logger.addHandler(rich_handler)

    # ── File handler (plain text) ─────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = "train_log.txt" if if_train else "test_log.txt"
        fh = logging.FileHandler(os.path.join(save_dir, fname), mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        )
        logger.addHandler(fh)

    return logger
