import logging
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from torch.cuda import amp
from processor.processor_au import AUEvaluator


def _unwrap_data_parallel(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _assert_finite_loss(loss, stage_name, epoch, iteration):
    if torch.isfinite(loss.detach()):
        return
    raise FloatingPointError(
        "{} produced a non-finite loss at epoch {}, iteration {}. {}".format(
            stage_name, epoch, iteration, _tensor_debug_summary(loss)
        )
    )


def _tensor_debug_summary(tensor):
    values = tensor.detach()
    total = values.numel()
    finite = torch.isfinite(values)
    finite_count = int(finite.sum().item())
    nan_count = int(torch.isnan(values).sum().item())
    posinf_count = int(torch.isposinf(values).sum().item())
    neginf_count = int(torch.isneginf(values).sum().item())
    if finite_count > 0:
        finite_values = values[finite]
        min_value = float(finite_values.min().item())
        max_value = float(finite_values.max().item())
        finite_range = "finite_min={:.4e}, finite_max={:.4e}".format(
            min_value, max_value
        )
    else:
        finite_range = "no finite values"
    return (
        "shape={}, dtype={}, finite={}/{}, nan={}, +inf={}, -inf={}, {}".format(
            tuple(values.shape),
            values.dtype,
            finite_count,
            total,
            nan_count,
            posinf_count,
            neginf_count,
            finite_range,
        )
    )


def _assert_finite_tensor(tensor, name, stage_name, epoch=None, iteration=None):
    values = tensor if torch.is_tensor(tensor) else torch.as_tensor(tensor)
    if torch.isfinite(values.detach()).all():
        return
    location = ""
    if epoch is not None and iteration is not None:
        location = " at epoch {}, iteration {}".format(epoch, iteration)
    raise FloatingPointError(
        "{} produced non-finite {}{}. {}".format(
            stage_name, name, location, _tensor_debug_summary(values)
        )
    )


def _assert_binary_targets(target, stage_name, epoch, iteration):
    _assert_finite_tensor(target, "targets", stage_name, epoch, iteration)
    if torch.logical_or(target == 0.0, target == 1.0).all():
        return
    raise ValueError(
        "{} received non-binary targets at epoch {}, iteration {}. {}".format(
            stage_name, epoch, iteration, _tensor_debug_summary(target)
        )
    )


def _as_float_tensor_list(values):
    if isinstance(values, (list, tuple)):
        return [value.float() for value in values]
    return values.float()


def _freeze_text_encoder(model):
    base_model = _unwrap_data_parallel(model)
    module = getattr(base_model, "text_encoder", None)
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(False)


def _evaluate_au_model(model, val_loader, device):
    evaluator = AUEvaluator()
    model.eval()
    evaluator.reset()
    for n_iter, (img, target, _, _, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            probs = model(img)
            _assert_finite_tensor(probs, "validation probabilities", "Evaluation")
            evaluator.update(probs, target)
    return evaluator.compute()


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
    text_model = _unwrap_data_parallel(model)
    _freeze_text_encoder(model)

    loss_meter = AverageMeter()
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    
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
        logger.info("Stage 1 LR: {:.2e}".format(optimizer.param_groups[0]["lr"]))
        
        for n_iter, (img, target, _, _, _) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            target = target.to(device).float() # Binary labels [B, 12]
            _assert_binary_targets(target, "Stage 1", epoch, n_iter + 1)
            
            with torch.amp.autocast('cuda', enabled=False):
                # Get Image Features
                with torch.no_grad():
                    image_features = model(x=img, get_image=True) # [B, 512]
                # Get Text Features for all 12 AUs
                text_features = text_model(get_text=True) # [12, 512]

            # Keep similarity and BCE math in fp32; fp16 normalize can easily
            # create NaNs when a feature norm gets very small.
            _assert_finite_tensor(
                image_features, "image features", "Stage 1", epoch, n_iter + 1
            )
            _assert_finite_tensor(
                text_features, "text features", "Stage 1", epoch, n_iter + 1
            )
            image_features = F.normalize(image_features.float(), dim=-1, eps=1e-6)
            text_features = F.normalize(text_features.float(), dim=-1, eps=1e-6)
            _assert_finite_tensor(
                image_features,
                "normalized image features",
                "Stage 1",
                epoch,
                n_iter + 1,
            )
            _assert_finite_tensor(
                text_features,
                "normalized text features",
                "Stage 1",
                epoch,
                n_iter + 1,
            )

            with torch.amp.autocast('cuda', enabled=False):
                # Compute logits: [B, 12]
                logits = (image_features @ text_features.t()) / temperature
                _assert_finite_tensor(
                    logits, "ITC logits", "Stage 1", epoch, n_iter + 1
                )
                loss = loss_fn_itc(logits, target)
            _assert_finite_loss(loss, "Stage 1", epoch, n_iter + 1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(text_model.prompt_learner.parameters(), 1.0)
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
    text_model = _unwrap_data_parallel(model)

    loss_meter = AverageMeter()
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    final_results = None
    best_disfa8_f1 = -1.0
    
    all_start_time = time.time()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()

        model.train()
        for n_iter, (img, target, _, _, _) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            target = target.to(device).float()
            _assert_binary_targets(target, "Stage 2", epoch, n_iter + 1)
            
            with torch.amp.autocast('cuda', enabled=True):
                # model returns [logits_list], [feat_list], img_feat_proj
                score, _, img_feat_proj = model(img)

            # Optional: Maintain Image-Text Alignment. This branch is frozen in
            # Stage 2, so keep it in fp32 and out of the autograd graph.
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                text_features = text_model(get_text=True)

            score = _as_float_tensor_list(score)
            score_tensors = score if isinstance(score, list) else [score]
            for index, score_tensor in enumerate(score_tensors):
                _assert_finite_tensor(
                    score_tensor,
                    "classifier logits[{}]".format(index),
                    "Stage 2",
                    epoch,
                    n_iter + 1,
                )

            img_feat_proj = F.normalize(img_feat_proj.float(), dim=-1, eps=1e-6)
            text_features = F.normalize(text_features.float(), dim=-1, eps=1e-6)
            with torch.amp.autocast('cuda', enabled=False):
                # Classification and BCE/ITC math stay in fp32. Weighted BCE can
                # overflow in fp16 before GradScaler has a chance to react.
                loss_cls = loss_fn(score, target)
                logits_itc = (img_feat_proj @ text_features.t()) / 0.07
                _assert_finite_tensor(
                    logits_itc,
                    "ITC logits",
                    "Stage 2",
                    epoch,
                    n_iter + 1,
                )
                loss_itc = F.binary_cross_entropy_with_logits(logits_itc, target)
                
                loss = loss_cls + 0.1 * loss_itc # Small weight for ITC
            _assert_finite_loss(loss, "Stage 2", epoch, n_iter + 1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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

        if epoch % eval_period == 0 or epoch == epochs:
            results = _evaluate_au_model(model, val_loader, device)
            final_results = results
            best_disfa8_f1 = max(best_disfa8_f1, results['disfa8_f1_macro'])
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("Avg F1: {:.4f}, Avg AUC: {:.4f}, Accuracy: {:.4f}"
                        .format(results['avg_f1'], results['avg_auc'], results['accuracy']))
            logger.info("DISFA-8 Avg F1: {:.4f}, Best DISFA-8 Avg F1: {:.4f}"
                        .format(results['disfa8_f1_macro'], best_disfa8_f1))

    if final_results is None:
        final_results = _evaluate_au_model(model, val_loader, device)

    logger.info("Total training time: {:.1f}s".format(time.time() - all_start_time))
    return final_results
