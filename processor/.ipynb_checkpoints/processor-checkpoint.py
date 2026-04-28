"""
processor/lightning_module_au.py
---------------------------------
PyTorch Lightning module thay thế processor/processor_au.py.
Bọc toàn bộ vòng lặp train / val, loss, optimizer và scheduler
vào một LightningModule duy nhất để dùng với Trainer (bao gồm multi-GPU).
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import lightning as L
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from loss.au_loss import WeightedBCELoss

try:
    from utils.rich_logger import setup_rich_logger, console
    logger = setup_rich_logger("transreid.train")
except Exception:
    logger = logging.getLogger("transreid.train")

AU_NAMES = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
REGION_NAMES = [
    "mắt trái", "mắt phải", "lông mày trái", "lông mày phải",
    "mũi", "môi trên", "môi dưới", "má", "hàm"
]


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class AULightningModule(L.LightningModule):
    """
    LightningModule cho bài toán AU Detection.

    Nhận:
        model          – build_transformer từ model/make_model.py
        loss_fn        – WeightedBCELoss từ loss/au_loss.py
        cfg            – YACS config (cfg_base)
        optimizer_fn   – callable(model) -> (optimizer, optimizer_center)
        scheduler_fn   – callable(optimizer) -> scheduler
    """

    def __init__(self, model, loss_fn, cfg, optimizer_fn, scheduler_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self._optimizer_fn = optimizer_fn
        self._scheduler_fn = scheduler_fn

        # Validation accumulators
        self._val_preds: list[np.ndarray] = []
        self._val_targets: list[np.ndarray] = []

        # Thống kê F1/AUC để báo cáo và tìm ngưỡng tối ưu
        self.best_thresholds = np.full(12, 0.5)
        self._last_au_f1: list[float] = [0.0] * 12
        self._last_au_auc: list[float] = [0.0] * 12

        # Stage management
        self.stage = 1

    def on_save_checkpoint(self, checkpoint):
        """Lưu lại stage hiện tại vào checkpoint."""
        checkpoint["training_stage"] = self.stage

    def on_load_checkpoint(self, checkpoint):
        """Khôi phục stage và cấu hình lại model tương ứng."""
        self.stage = checkpoint.get("training_stage", 1)
        # Đảm bảo model được cấu hình đúng với stage vừa load
        self.set_stage(self.stage)
        logger.info(f">>> Resumed from checkpoint at Stage {self.stage}")

    def set_stage(self, stage):
        """Chuyển đổi giai đoạn huấn luyện."""
        self.stage = stage
        if hasattr(self.model, 'set_train_stage'):
            self.model.set_train_stage(stage)
        else:
            # Fallback nếu model được bọc TRONG một module khác
            self.model.module.set_train_stage(stage) if hasattr(self.model, 'module') else None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, img, landmarks=None):
        return self.model(img, landmarks=landmarks)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        img, target, _, _, _, landmarks = batch
        landmarks = landmarks.to(self.device)

        # score is list of scores, feats is list of feats
        score, feats = self.model(img, landmarks=landmarks)

        # Loss tính ngoài autocast (float32) để tránh overflow
        with torch.amp.autocast("cuda", enabled=False):
            loss = self.loss_fn(score, target.float())
            
            # --- Anatomical Penalty for Stage 2 ---
            if self.stage == 2 and len(feats) > 3:
                # feats[3] is the attn_penalty returned from build_transformer.forward
                attn_penalty = feats[3]
                # Use weight from config (default 1.0)
                penalty_weight = self.cfg.SOLVER.STAGE2.ATTN_PENALTY_WEIGHT
                loss = loss + penalty_weight * attn_penalty
                self.log("train/attn_penalty", attn_penalty, on_step=True, on_epoch=True, sync_dist=True)

        # Guard NaN / huge loss
        if torch.isnan(loss) or loss.item() > 1000.0:
            logger.error(
                f"[Epoch {self.current_epoch} step {batch_idx}] "
                f"Loss anomaly: {loss.item():.4f}"
            )
            for i, s in enumerate(score if isinstance(score, list) else [score]):
                logger.error(
                    f"  score[{i}] max={s.max().item():.4f} "
                    f"min={s.min().item():.4f}"
                )
            if torch.isnan(loss):
                raise ValueError("NaN loss detected – training stopped. Check logs.")
        
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    # Gradient clipping (Lightning hook, gọi sau backward)
    # ------------------------------------------------------------------

    def on_before_optimizer_step(self, optimizer):
        # Clip gradient để tránh exploding
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        img, target, _, _, _, landmarks = batch
        landmarks = landmarks.to(self.device)

        # Inference mode: model trả về sigmoid probabilities
        # Lấy focus mapping cho batch đầu tiên của validation để debug
        if batch_idx == 0 and self.current_epoch % 1 == 0:
            probs, focus = self.model(img, landmarks=landmarks, return_attn=True)
            self._print_au_focus(focus[0]) # In mẫu đầu tiên trong batch
        else:
            probs = self.model(img, landmarks=landmarks)
            
        # Tích lũy để tính metric cuối epoch
        self._val_preds.append(probs.cpu().float().numpy())
        self._val_targets.append(target.cpu().float().numpy())

    def _print_au_focus(self, focus_matrix):
        """
        focus_matrix: [12, 9] (tensor)
        In ra vùng mà mỗi AU đang tập trung nhất.
        """
        try:
            # Chuyển sang numpy để xử lý
            focus_np = focus_matrix.detach().cpu().numpy() # [12, 9]
            
            console.print("\n[bold cyan]>>> AU-to-Region Attention Focus (Epoch {})[/bold cyan]".format(self.current_epoch))
            from rich.table import Table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("AU", style="dim", width=12)
            table.add_column("Primary Focus Region", justify="left")
            table.add_column("Confidence", justify="right")

            for i, au_id in enumerate(AU_NAMES):
                region_idx = np.argmax(focus_np[i])
                confidence = focus_np[i][region_idx]
                region_name = REGION_NAMES[region_idx]
                table.add_row(f"AU{au_id}", region_name, f"{confidence:.4f}")
            
            console.print(table)
        except Exception as e:
            logger.warning(f"Could not print AU focus: {e}")

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return

        preds = np.concatenate(self._val_preds, axis=0)
        targets = np.concatenate(self._val_targets, axis=0)
        self._val_preds.clear()
        self._val_targets.clear()

        # --- Tối ưu hóa Threshold cho từng AU ---
        best_au_f1 = np.zeros(12)
        best_thresholds = np.zeros(12)
        
        # Thử các ngưỡng từ 0.05 đến 0.95
        threshold_candidates = np.linspace(0.05, 0.95, 19)
        
        for i in range(12):
            au_targets = targets[:, i]
            au_probs = preds[:, i]
            
            # Skip nếu không có mẫu hợp lệ
            valid_mask = (au_targets >= 0) & (~np.isnan(au_targets))
            if not np.any(valid_mask):
                continue
            
            y_true = au_targets[valid_mask]
            y_prob = au_probs[valid_mask]
            
            best_f1 = -1.0
            best_t = 0.5
            for t in threshold_candidates:
                y_pred = (y_prob > t).astype(int)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            
            best_au_f1[i] = best_f1
            best_thresholds[i] = best_t

        self.best_thresholds = best_thresholds
        self._last_au_f1 = [float(v) for v in best_au_f1]
        avg_f1 = float(np.mean(best_au_f1))

        # --- Tính AUC ---
        au_auc = np.zeros(12)
        for i in range(12):
            try:
                valid_mask = (targets[:, i] >= 0) & (~np.isnan(targets[:, i]))
                au_auc[i] = roc_auc_score(targets[valid_mask, i], preds[valid_mask, i])
            except ValueError:
                au_auc[i] = 0.5
        self._last_au_auc = [float(v) for v in au_auc]
        avg_auc = float(np.mean(au_auc))

        # --- In bảng thống kê chi tiết ---
        console.print(f"\n[bold green]>>> Detailed AU Evaluation (Epoch {self.current_epoch})[/bold green]")
        from rich.table import Table
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("AU", style="dim")
        table.add_column("Pos Ratio", justify="right")
        table.add_column("Pred-Pos Ratio", justify="right")
        table.add_column("Best Thresh", justify="right", style="cyan")
        table.add_column("F1", justify="right", style="bold green")
        table.add_column("AUC", justify="right", style="bold magenta")

        for i, au_id in enumerate(AU_NAMES):
            pos_ratio = np.mean(targets[:, i][targets[:, i] >= 0])
            pred_pos_ratio = np.mean((preds[:, i] > best_thresholds[i]).astype(float))
            table.add_row(
                f"AU{au_id}",
                f"{pos_ratio:.2%}",
                f"{pred_pos_ratio:.2%}",
                f"{best_thresholds[i]:.2f}",
                f"{best_au_f1[i]:.4f}",
                f"{au_auc[i]:.4f}"
            )
        console.print(table)

        self.log("val/avg_f1",  avg_f1,  prog_bar=True,  sync_dist=True)
        self.log("val/avg_auc", avg_auc, prog_bar=True,  sync_dist=True)

    # ------------------------------------------------------------------
    # Optimizer & Scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # Truyền stage vào hàm factory để build đúng optimizer cho giai đoạn đó
        optimizer, _ = self._optimizer_fn(self.model, stage=self.stage)
        scheduler = self._scheduler_fn(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
