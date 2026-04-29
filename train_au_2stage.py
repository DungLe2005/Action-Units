import os
import argparse
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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AU Detection Two-Stage Training")
    parser.add_argument(
        "--config_file", default="configs/au/vit_base_au_2stage.yaml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.NAMES = 'disfa' # Ensure disfa for AU
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # 1. Load Data
    train_loader, val_loader, num_aus = make_au_dataloader(cfg)

    # 2. Build Model
    model = make_model(cfg, num_class=num_aus, camera_num=1, view_num=1)

    # 3. Build Loss
    loss_func, center_criterion = make_loss(cfg, num_classes=num_aus)

    # -------------------------------------------------------------------------
    # STAGE 1: Image-Text Alignment
    # -------------------------------------------------------------------------
    logger.info("Starting Stage 1...")
    optimizer_1stage = make_optimizer_1stage(cfg, model)
    scheduler_1stage = create_scheduler(optimizer_1stage, 
                                        num_epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS, 
                                        lr_min = cfg.SOLVER.STAGE1.LR_MIN, 
                                        warmup_lr_init = cfg.SOLVER.STAGE1.WARMUP_LR_INIT, 
                                        warmup_t = cfg.SOLVER.STAGE1.WARMUP_EPOCHS, 
                                        noise_range = None)

    do_train_stage1(
        cfg,
        model,
        train_loader,
        optimizer_1stage,
        scheduler_1stage,
        args.local_rank
    )

    # -------------------------------------------------------------------------
    # STAGE 2: Full Fine-tuning
    # -------------------------------------------------------------------------
    logger.info("Starting Stage 2...")
    # Optional: Load best Stage 1 checkpoint if needed
    # model.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_au_stage1_10.pth')))

    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
    scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, 
                                          cfg.SOLVER.STAGE2.STEPS, 
                                          cfg.SOLVER.STAGE2.GAMMA, 
                                          cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                                          cfg.SOLVER.STAGE2.WARMUP_ITERS, 
                                          cfg.SOLVER.STAGE2.WARMUP_METHOD)

    do_train_stage2(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer_2stage,
        scheduler_2stage,
        loss_func,
        args.local_rank
    )
