import os
import argparse
import random
import torch
import numpy as np
from config import cfg_base as cfg
from utils.logger import setup_logger
from datasets.make_dataloader import make_au_dataloader
from model.make_model import make_model
from solver.make_optimizer import make_optimizer
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_au import do_train_au

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AU Detection Training")
    parser.add_argument(
        "--config_file", default="configs/au/vit_base_au.yaml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Ensure dataset is DISFA for correctly choosing loss
    cfg.DATASETS.NAMES = 'disfa'
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
    # num_aus will be used if names=='disfa' in make_model
    model = make_model(cfg, num_class=num_aus, camera_num=1, view_num=1)

    # 3. Build Loss
    loss_func, center_criterion = make_loss(cfg, num_classes=num_aus)

    # 4. Build Optimizer
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    # 5. Build Scheduler
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    # 6. Train
    do_train_au(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        args.local_rank
    )
