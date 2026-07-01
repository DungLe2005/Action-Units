import os
import argparse
import logging
import random
import torch
import numpy as np
from config import cfg_base as cfg
from utils.logger import setup_logger
from datasets.make_dataloader import make_au_dataloader
from model.make_model import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_au_2stage import do_train_stage1, do_train_stage2
from utils.au_fold_report import flatten_fold_metrics, write_fold_reports


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _reset_logger_handlers():
    logger = logging.getLogger("transreid")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _format_cuda_visible_devices(device_id):
    if isinstance(device_id, (list, tuple)):
        return ",".join(str(gpu_id) for gpu_id in device_id)
    return str(device_id)


def _log_cuda_setup(logger):
    logger.info(
        "CUDA_VISIBLE_DEVICES={}".format(
            os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        )
    )
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available; training will run on CPU.")
        return

    device_count = torch.cuda.device_count()
    device_summaries = []
    for device_index in range(device_count):
        props = torch.cuda.get_device_properties(device_index)
        total_gb = props.total_memory / (1024 ** 3)
        device_summaries.append(
            "{}:{} ({:.1f}GB)".format(device_index, props.name, total_gb)
        )
    logger.info(
        "PyTorch visible CUDA devices: {} [{}]".format(
            device_count, "; ".join(device_summaries)
        )
    )


def _make_fold_cfg(base_cfg, fold_idx, output_dir):
    fold_cfg = base_cfg.clone()
    fold_cfg.defrost()
    fold_cfg.OUTPUT_DIR = os.path.join(output_dir, f"fold_{fold_idx}")
    fold_cfg.freeze()
    return fold_cfg


def _make_stage1_scheduler(cfg, optimizer):
    return create_scheduler(
        optimizer,
        num_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS,
        lr_min=cfg.SOLVER.STAGE1.LR_MIN,
        warmup_lr_init=cfg.SOLVER.STAGE1.WARMUP_LR_INIT,
        warmup_t=cfg.SOLVER.STAGE1.WARMUP_EPOCHS,
        noise_range=None,
    )


def _make_stage2_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STAGE2.STEPS,
        cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR,
        cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD,
    )


def run_fold(cfg, args, fold_idx):
    set_seed(cfg.SOLVER.SEED)

    if cfg.OUTPUT_DIR and not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    _reset_logger_handlers()
    logger = setup_logger("transreid", cfg.OUTPUT_DIR, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)
    _log_cuda_setup(logger)
    logger.info("Starting DISFA subject-exclusive fold {}".format(fold_idx))

    train_loader, val_loader, num_aus, pos_weight, fold_info = make_au_dataloader(
        cfg, fold_idx=fold_idx
    )
    logger.info(
        "Effective batch sizes - train: {}, val: {}, num_workers: {}".format(
            train_loader.batch_size, val_loader.batch_size, train_loader.num_workers
        )
    )
    logger.info("Fold info: {}".format(fold_info))

    model = make_model(cfg, num_class=num_aus, camera_num=1, view_num=1)
    if bool(getattr(cfg.SOLVER.STAGE2, "INIT_HEAD_BIAS_FROM_PRIOR", True)):
        model.init_au_head_bias_from_pos_weight(pos_weight)
        logger.info("Initialized AU head biases from train-split class priors")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_func, center_criterion = make_loss(
        cfg, num_classes=num_aus, pos_weight=pos_weight, device=device
    )

    if args.resume:
        model.load_param(args.resume)
        logger.info(f"Resuming from {args.resume}")

    if not args.skip_stage1:
        logger.info("Starting Stage 1...")
        optimizer_1stage = make_optimizer_1stage(cfg, model)
        scheduler_1stage = _make_stage1_scheduler(cfg, optimizer_1stage)
        do_train_stage1(
            cfg,
            model,
            train_loader,
            optimizer_1stage,
            scheduler_1stage,
            args.local_rank,
            val_loader=val_loader,
        )
    else:
        logger.info("Skipping Stage 1 as requested.")

    logger.info("Starting Stage 2...")
    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(
        cfg, model, center_criterion
    )
    scheduler_2stage = _make_stage2_scheduler(cfg, optimizer_2stage)
    metrics = do_train_stage2(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer_2stage,
        scheduler_2stage,
        loss_func,
        args.local_rank,
    )
    return {
        "fold_info": fold_info,
        "metrics": metrics,
        "row": flatten_fold_metrics(metrics, fold_info),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AU Detection Two-Stage Training")
    parser.add_argument(
        "--config_file",
        default="configs/au/vit_base_au_2stage.yaml",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument(
        "--resume", default="", help="path to checkpoint to resume", type=str
    )
    parser.add_argument(
        "--skip_stage1", action="store_true", help="skip stage 1 and start from stage 2"
    )
    parser.add_argument("--fold_idx", default=0, type=int, help="DISFA fold index")
    parser.add_argument(
        "--all_folds",
        action="store_true",
        help="run all 3 subject-exclusive DISFA folds and export CSV/JSON summary",
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.NAMES = "disfa"  # Ensure disfa for AU
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = _format_cuda_visible_devices(
        cfg.MODEL.DEVICE_ID
    )

    if args.all_folds:
        if args.resume:
            raise ValueError("--resume is only supported for a single fold run")
        base_output_dir = cfg.OUTPUT_DIR
        if base_output_dir and not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)
        fold_records = []
        for fold_idx in range(3):
            fold_cfg = _make_fold_cfg(cfg, fold_idx, base_output_dir)
            fold_records.append(run_fold(fold_cfg, args, fold_idx))
        report_info = write_fold_reports(base_output_dir, fold_records)
        print("Fold metrics CSV: {}".format(report_info["csv_path"]))
        print("Fold metrics JSON: {}".format(report_info["json_path"]))
    else:
        run_fold(cfg, args, args.fold_idx)
