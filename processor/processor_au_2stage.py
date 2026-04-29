import logging
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from torch.cuda import amp
from processor.processor_au import AUEvaluator

def do_train_stage1(cfg,
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD

    logger = logging.getLogger("transreid.train")
    logger.info('Start AU Training Stage 1 (Image-Text Alignment)')
    
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    # Multi-label Contrastive Loss using BCE
    loss_fn_itc = nn.BCEWithLogitsLoss()
    temperature = 0.07 # Standard CLIP temperature

    all_start_time = time.time()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        model.train()
        
        # In Stage 1, we only want to optimize the prompt learner
        # The optimizer passed here should already be configured for that
        
        for n_iter, (img, target, _, _, _) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            target = target.to(device) # Binary labels [B, 12]
            
            with torch.amp.autocast('cuda', enabled=True):
                # Get Image Features
                image_features = model(x=img, get_image=True) # [B, 512]
                # Get Text Features for all 12 AUs
                text_features = model(get_text=True) # [12, 512]
                
                # Normalize features
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Compute logits: [B, 12]
                logits = (image_features @ text_features.t()) / temperature
                loss = loss_fn_itc(logits, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}"
                            .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg))

        scheduler.step(epoch)
        logger.info("Epoch {} done. Time: {:.1f}s".format(epoch, time.time() - start_time))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_au_stage1_{}.pth'.format(epoch)))

    logger.info("Stage 1 training time: {:.1f}s".format(time.time() - all_start_time))


def do_train_stage2(cfg,
                    model,
                    train_loader,
                    val_loader,
                    optimizer,
                    scheduler,
                    loss_fn,
                    local_rank):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('Start AU Training Stage 2 (Fine-tuning)')
    
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    evaluator = AUEvaluator()
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    all_start_time = time.time()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        evaluator.reset()

        model.train()
        for n_iter, (img, target, _, _, _) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            target = target.to(device)
            
            with torch.amp.autocast('cuda', enabled=True):
                # model returns [logits_list], [feat_list], img_feat_proj
                score, _, img_feat_proj = model(img)
                
                # Classification Loss
                loss_cls = loss_fn(score, target)
                
                # Optional: Maintain Image-Text Alignment
                text_features = model(get_text=True)
                img_feat_proj = F.normalize(img_feat_proj, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                logits_itc = (img_feat_proj @ text_features.t()) / 0.07
                loss_itc = F.binary_cross_entropy_with_logits(logits_itc, target)
                
                loss = loss_cls + 0.1 * loss_itc # Small weight for ITC

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, scheduler.get_lr()[0]))

        scheduler.step()
        logger.info("Epoch {} done. Time: {:.1f}s".format(epoch, time.time() - start_time))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_au_stage2_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            evaluator.reset()
            for n_iter, (img, target, _, _, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    probs = model(img)
                    evaluator.update(probs, target)
            
            results = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("Avg F1: {:.4f}, Avg AUC: {:.4f}, Accuracy: {:.4f}"
                        .format(results['avg_f1'], results['avg_auc'], results['accuracy']))

    logger.info("Total training time: {:.1f}s".format(time.time() - all_start_time))
