"""
train_au_lightning.py
-----------------------
Entry point thay thế train_au.py, dùng PyTorch Lightning Trainer
để train AU Detection trên 1 hoặc nhiều GPU.

Chạy từ terminal:
    (1 GPU)  python train_au_lightning.py
    (2 GPU)  python train_au_lightning.py --devices 2
    (Jupyter) chỉ cần import và gọi hàm run_training() ở cuối file.
"""

import os
import argparse
import random
import sys
import multiprocessing as mp

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.strategies import DDPStrategy

from config import cfg_base
from utils.rich_logger import setup_rich_logger, console
from utils.rich_callback import RichTrainingCallback
from datasets.make_dataloader import make_au_dataloader
from model.make_model import make_model
from solver.make_optimizer import make_optimizer
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_au import AULightningModule
from functools import partial
import inspect



# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def is_interactive_env():
    """Kiểm tra xem code có đang chạy trong môi trường interactive (Jupyter/IPython) không."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        return shell_name in ("ZMQInteractiveShell", "TerminalInteractiveShell")
    except (NameError, ImportError, AttributeError):
        return hasattr(sys, "ps1") or "ipykernel" in sys.modules


# ---------------------------------------------------------------------------
# Core training function (có thể gọi trực tiếp từ Jupyter)
# ---------------------------------------------------------------------------

# --- Optimizer / Scheduler factories ---
def _optimizer_fn(m, cfg, stage=1):
    center = _dummy_center_criterion()

    has_stage = "stage" in inspect.signature(make_optimizer).parameters
    if has_stage:
        return make_optimizer(cfg, m, center, stage=stage)

    # fallback cho make_optimizer cũ (không có stage)
    stage_cfg = cfg.clone()
    stage_cfg.defrost()
    if stage == 1:
        s = stage_cfg.SOLVER.STAGE1
    else:
        s = stage_cfg.SOLVER.STAGE2

    stage_cfg.SOLVER.BASE_LR = s.BASE_LR
    stage_cfg.SOLVER.WEIGHT_DECAY = s.WEIGHT_DECAY
    stage_cfg.SOLVER.WEIGHT_DECAY_BIAS = s.WEIGHT_DECAY_BIAS
    stage_cfg.SOLVER.OPTIMIZER_NAME = s.OPTIMIZER_NAME
    if hasattr(s, "LARGE_FC_LR"):
        stage_cfg.SOLVER.LARGE_FC_LR = s.LARGE_FC_LR
    stage_cfg.freeze()

    return make_optimizer(stage_cfg, m, center)

def _scheduler_fn(opt, cfg):
    return WarmupMultiStepLR(
        opt,
        cfg.SOLVER.STAGE2.STEPS,
        cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR,
        cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD,
    )

def find_latest_checkpoint(output_dir):
    """Tìm checkpoint mới nhất (ưu tiên last.ckpt)."""
    if not os.path.exists(output_dir):
        return None
    
    # Ưu tiên last.ckpt
    last_path = os.path.join(output_dir, "last.ckpt")
    if os.path.exists(last_path):
        return last_path
        
    # Nếu không có last.ckpt, tìm file .ckpt mới nhất theo thời gian sửa đổi
    checkpoints = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".ckpt")]
    if not checkpoints:
        return None
        
    return max(checkpoints, key=os.path.getmtime)

def get_stage_from_checkpoint(ckpt_path):
    """Đọc stage từ file checkpoint mà không cần load toàn bộ model."""
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        return checkpoint.get("training_stage", 1)
    except Exception:
        return 1

def run_training(
    config_file: str = "configs/au/vit_base_au.yaml",
    devices: int | list[int] = "auto",
    precision: str = "16-mixed",
    extra_opts: list[str] | None = None,
    resume: bool = False,
    checkpoint_path: str | None = None,
):
    """
    Khởi động quá trình training.

    Args:
        config_file: Đường dẫn đến file YAML config.
        devices    : Số GPU (int) hoặc danh sách GPU IDs, ví dụ 2 hoặc [0, 1].
                     Dùng "auto" để tự phát hiện.
        precision  : "16-mixed" (AMP, khuyên dùng), "32", "bf16-mixed".
        extra_opts : Danh sách opts bổ sung cho cfg (kiểu YACS).
        resume     : Tự động tìm checkpoint cuối cùng để tiếp tục.
        checkpoint_path: Đường dẫn cụ thể đến file .ckpt (ghi đè resume).
    """
    
    # CUDA + multiprocessing trong notebook cần 'spawn' để tránh lỗi re-init CUDA.
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    cfg = cfg_base.clone()
    # --- Config ---
    cfg.defrost()
    if config_file:
        cfg.merge_from_file(config_file)
    if extra_opts:
        cfg.merge_from_list(extra_opts)
    cfg.DATASETS.NAMES = "disfa"
    if is_interactive_env():
        # Tránh lỗi CUDA re-init từ DataLoader worker subprocess trong notebook.
        cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    # --- Output dir ---
    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_rich_logger("transreid", output_dir, if_train=True)
    logger.info(f"Config file: {config_file}")
    logger.info(f"Output dir : {output_dir}")
    logger.info(f"Devices    : {devices}")
    logger.info(f"Precision  : {precision}")

    # --- Data ---
    train_loader, val_loader, num_aus = make_au_dataloader(cfg)

    # --- Model ---
    model = make_model(cfg, num_class=num_aus, camera_num=1, view_num=1)

    # --- Loss ---
    loss_fn, _ = make_loss(cfg, num_classes=num_aus)

    optimizer_fn = partial(_optimizer_fn, cfg=cfg)
    scheduler_fn = partial(_scheduler_fn, cfg=cfg)

    # --- Lightning Module ---
    pl_model = AULightningModule(
        model=model,
        loss_fn=loss_fn,
        cfg=cfg,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
    )

    # --- Callbacks ---
    # --- Callbacks ---
    rich_cb = RichTrainingCallback()
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Tách biệt checkpoint cho Stage 1 và Stage 2 để tránh bị Stage 2 ghi đè/xóa mất
    checkpoint_cb_stage1 = ModelCheckpoint(
        dirpath=output_dir,
        # Đổi {epoch:02d} thành {epoch:02.0f}
        filename=cfg.MODEL.NAME + "_au_stage1_{epoch:02.0f}_{val/avg_f1:.4f}",
        monitor="val/avg_f1",
        mode="max",
        save_top_k=2,
        every_n_epochs=cfg.SOLVER.EVAL_PERIOD,
    )
    checkpoint_cb_stage2 = ModelCheckpoint(
        dirpath=output_dir,
        # Đổi {epoch:02d} thành {epoch:02.0f}
        filename=cfg.MODEL.NAME + "_au_stage2_{epoch:02.0f}_{val/avg_f1:.4f}",
        monitor="val/avg_f1",
        mode="max",
        save_top_k=3,
        every_n_epochs=cfg.SOLVER.EVAL_PERIOD,
        save_last=True,
    )

    # --- Strategy: DDP khi nhiều GPU ---
    if devices == "auto":
        num_devices = max(torch.cuda.device_count(), 1)
    elif isinstance(devices, int):
        num_devices = devices
    else:
        num_devices = len(devices)

    use_ddp = num_devices > 1

    if use_ddp:
        if is_interactive_env():
            # Notebook yêu cầu chiến lược đặc biệt để spawn process
            strategy = "ddp_notebook"
        else:
            strategy = DDPStrategy(find_unused_parameters=True, start_method="spawn")
    else:
        strategy = "auto"
    logger.info(f"Strategy   : {strategy}")

    def build_strategy(use_ddp: bool):
        if use_ddp:
            return DDPStrategy(find_unused_parameters=True, start_method="spawn")
        return "auto"

    # --- Resume Logic ---
    ckpt_to_resume = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt_to_resume = checkpoint_path
        logger.info(f"Using explicitly provided checkpoint: {ckpt_to_resume}")
    elif resume:
        ckpt_to_resume = find_latest_checkpoint(output_dir)
        if ckpt_to_resume:
            logger.info(f"Auto-resuming from latest checkpoint: {ckpt_to_resume}")
        else:
            logger.warning("Resume flag set but no checkpoint found in output directory. Starting from scratch.")

    start_stage = 1
    if ckpt_to_resume:
        start_stage = get_stage_from_checkpoint(ckpt_to_resume)
        logger.info(f"Checkpoint indicates Stage {start_stage}")

    # --- Stage 1 Training ---
    if start_stage <= 1:
        pl_model.set_stage(1)
        trainer_stage1 = L.Trainer(
            max_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS,
            devices=devices,
            strategy=build_strategy(use_ddp),
            precision=precision,
            callbacks=[rich_cb, checkpoint_cb_stage1, lr_monitor],
            log_every_n_steps=cfg.SOLVER.STAGE1.LOG_PERIOD,
            check_val_every_n_epoch=cfg.SOLVER.STAGE1.EVAL_PERIOD,
            enable_progress_bar=False,
            default_root_dir=output_dir,
        )
        
        logger.info(">>> Starting Stage 1 Training (Frozen Backbone)...")
        # Passing ckpt_path to fit handles resuming optimizer/epoch
        trainer_stage1.fit(pl_model, train_loader, val_loader, ckpt_path=ckpt_to_resume if start_stage == 1 else None)
        
        # After Stage 1, we clear ckpt_to_resume so Stage 2 starts fresh unless we just resumed IN Stage 2
        ckpt_to_resume = None 

    # --- Stage 2 Training ---
    if start_stage <= 2:
        pl_model.set_stage(2)
        trainer_stage2 = L.Trainer(
            max_epochs=cfg.SOLVER.STAGE2.MAX_EPOCHS,
            devices=devices,
            strategy=build_strategy(use_ddp),
            precision=precision,
            callbacks=[rich_cb, checkpoint_cb_stage2, lr_monitor],
            log_every_n_steps=cfg.SOLVER.STAGE2.LOG_PERIOD,
            check_val_every_n_epoch=cfg.SOLVER.STAGE2.EVAL_PERIOD,
            enable_progress_bar=False,
            default_root_dir=output_dir,
        )

        logger.info(">>> Starting Stage 2 Training (Fine-tuning with LoRA)...")
        # Resume Stage 2 if specified or if we just jumped to stage 2 from checkpoint
        trainer_stage2.fit(pl_model, train_loader, val_loader, ckpt_path=ckpt_to_resume if start_stage == 2 else None)
    
    logger.info("All Training Stages complete.")

    return pl_model


# ---------------------------------------------------------------------------
# Dummy center criterion (chỉ cần để make_optimizer không lỗi)
# ---------------------------------------------------------------------------

def _dummy_center_criterion():
    """make_optimizer yêu cầu center_criterion dù AU không dùng center loss."""
    from loss.center_loss import CenterLoss
    return CenterLoss(num_classes=12, feat_dim=2048, use_gpu=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AU Detection Training (Lightning)")
    parser.add_argument(
        "--config_file",
        default="configs/au/vit_base_au.yaml",
        type=str,
        help="path to config file",
    )
    parser.add_argument(
        "--devices",
        default=2,
        type=int,
        help="number of GPUs to use (default: 2)",
    )
    parser.add_argument(
        "--precision",
        default="16-mixed",
        type=str,
        choices=["16-mixed", "bf16-mixed", "32"],
        help="floating point precision",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="auto-resume training from the latest checkpoint in output_dir",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="path to a specific .ckpt file to resume from",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="optional extra YACS config overrides",
        default=None,
    )
    args = parser.parse_args()

    run_training(
        config_file=args.config_file,
        devices=args.devices,
        precision=args.precision,
        extra_opts=args.opts if args.opts else None,
        resume=args.resume,
        checkpoint_path=args.checkpoint,
    )
