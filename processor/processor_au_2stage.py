import logging
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from processor.processor_au import AUEvaluator
from tqdm import tqdm
from utils.au_training_history import history_row, write_stage2_history


PRIMARY_STAGE2_METRIC = "disfa8_f1_macro"


def _unwrap_data_parallel(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _save_model_state(model, checkpoint_path):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(_unwrap_data_parallel(model).state_dict(), checkpoint_path)


def _current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def _reset_cuda_peak_memory():
    if not torch.cuda.is_available():
        return
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)


def _cuda_memory_summary():
    if not torch.cuda.is_available():
        return "n/a"
    parts = []
    for device_index in range(torch.cuda.device_count()):
        peak_gb = torch.cuda.max_memory_allocated(device_index) / (1024 ** 3)
        total_gb = (
            torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
        )
        parts.append("{}:{:.1f}/{:.0f}G".format(device_index, peak_gb, total_gb))
    return " ".join(parts)


def _samples_per_second(samples_seen, start_time):
    elapsed = max(time.time() - start_time, 1e-6)
    return float(samples_seen) / elapsed


def _prepare_cuda_model(model, logger, stage_name, batch_size):
    model.to("cuda")
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        per_gpu_batch = int(math.ceil(float(batch_size) / float(gpu_count)))
        logger.info(
            "{} using DataParallel on {} GPUs. Global train batch={}, "
            "approx per-GPU batch={}".format(
                stage_name, gpu_count, batch_size, per_gpu_batch
            )
        )
        return nn.DataParallel(model)
    logger.info("{} using single CUDA device. Train batch={}".format(stage_name, batch_size))
    return model


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


def _is_finite_tensor(tensor):
    values = tensor if torch.is_tensor(tensor) else torch.as_tensor(tensor)
    return bool(torch.isfinite(values.detach()).all().item())


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


def _au_probabilities_from_logits(output):
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, list):
        output = torch.stack([value.float() for value in output], dim=0).mean(dim=0)
    logits = output.float()
    if logits.ndim != 2 or logits.shape[1] != 12:
        raise ValueError(
            "AU evaluation expected AU logits shape [B, 12], got {}".format(
                tuple(logits.shape)
            )
        )
    return torch.sigmoid(logits)


def _freeze_text_encoder(model):
    base_model = _unwrap_data_parallel(model)
    module = getattr(base_model, "text_encoder", None)
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(False)


def _evaluate_au_model(model, val_loader, device, desc="Eval AU"):
    evaluator = AUEvaluator()
    model.eval()
    evaluator.reset()
    progress = tqdm(
        val_loader,
        desc=desc,
        total=len(val_loader),
        dynamic_ncols=True,
        leave=False,
    )
    for img, target, _, _, _ in progress:
        with torch.no_grad():
            img = img.to(device)
            logits = model(img, return_au_logits=True)
            probs = _au_probabilities_from_logits(logits)
            _assert_finite_tensor(probs, "validation probabilities", "Evaluation")
            evaluator.update(probs, target)
    return evaluator.compute()


def _run_stage2_eval(
    cfg,
    model,
    val_loader,
    device,
    epoch,
    train_loss,
    lr,
    best_metric,
    history_records,
    itc_enabled,
    logger,
):
    results = _evaluate_au_model(
        model, val_loader, device, desc="Eval Stage2 E{}".format(epoch)
    )
    current_metric = float(results.get(PRIMARY_STAGE2_METRIC, float("nan")))
    is_best = math.isfinite(current_metric) and current_metric > best_metric

    if is_best:
        best_metric = current_metric
        best_checkpoint = os.path.join(
            cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_au_stage2_best.pth"
        )
        _save_model_state(model, best_checkpoint)
        logger.info(
            "Saved best Stage 2 checkpoint: {} ({}={:.4f})".format(
                best_checkpoint, PRIMARY_STAGE2_METRIC, best_metric
            )
        )

    history_records.append(
        history_row(
            epoch=epoch,
            train_loss=train_loss,
            lr=lr,
            metrics=results,
            best_metric=best_metric,
            is_best=is_best,
            itc_enabled=itc_enabled,
        )
    )
    history_paths = write_stage2_history(cfg.OUTPUT_DIR, history_records)

    logger.info("Validation Results - Epoch: {}".format(epoch))
    logger.info(
        "Avg F1: {:.4f}, Avg AUC: {:.4f}, Accuracy: {:.4f}".format(
            results["avg_f1"], results["avg_auc"], results["accuracy"]
        )
    )
    logger.info(
        "DISFA-8 Avg F1: {:.4f}, Best DISFA-8 Avg F1: {:.4f}".format(
            current_metric, best_metric
        )
    )
    logger.info(
        "Stage 2 history: CSV={}, JSON={}, Plot={}".format(
            history_paths["csv_path"],
            history_paths["json_path"],
            history_paths["plot_path"] or "matplotlib unavailable",
        )
    )
    return results, best_metric


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
    
    model = _prepare_cuda_model(model, logger, "Stage 1", train_loader.batch_size)
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
        _reset_cuda_peak_memory()
        samples_seen = 0
        loss_meter.reset()
        model.train()
        
        # In Stage 1, we only want to optimize the prompt learner
        # The optimizer passed here should already be configured for that
        logger.info("Stage 1 LR: {:.2e}".format(optimizer.param_groups[0]["lr"]))
        
        progress = tqdm(
            train_loader,
            desc="Stage1 E{}/{}".format(epoch, epochs),
            total=len(train_loader),
            dynamic_ncols=True,
            leave=False,
        )
        for iteration, (img, target, _, _, _) in enumerate(progress, start=1):
            optimizer.zero_grad()
            img = img.to(device)
            target = target.to(device).float() # Binary labels [B, 12]
            _assert_binary_targets(target, "Stage 1", epoch, iteration)
            
            with torch.amp.autocast('cuda', enabled=False):
                # Get Image Features
                with torch.no_grad():
                    image_features = model(x=img, get_image=True) # [B, 512]
                # Get Text Features for all 12 AUs
                text_features = text_model(get_text=True) # [12, 512]

            # Keep similarity and BCE math in fp32; fp16 normalize can easily
            # create NaNs when a feature norm gets very small.
            _assert_finite_tensor(
                image_features, "image features", "Stage 1", epoch, iteration
            )
            _assert_finite_tensor(
                text_features, "text features", "Stage 1", epoch, iteration
            )
            image_features = F.normalize(image_features.float(), dim=-1, eps=1e-6)
            text_features = F.normalize(text_features.float(), dim=-1, eps=1e-6)
            _assert_finite_tensor(
                image_features,
                "normalized image features",
                "Stage 1",
                epoch,
                iteration,
            )
            _assert_finite_tensor(
                text_features,
                "normalized text features",
                "Stage 1",
                epoch,
                iteration,
            )

            with torch.amp.autocast('cuda', enabled=False):
                # Compute logits: [B, 12]
                logits = (image_features @ text_features.t()) / temperature
                _assert_finite_tensor(
                    logits, "ITC logits", "Stage 1", epoch, iteration
                )
                loss = loss_fn_itc(logits, target)
            _assert_finite_loss(loss, "Stage 1", epoch, iteration)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(text_model.prompt_learner.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])
            samples_seen += img.shape[0]
            imgs_per_sec = _samples_per_second(samples_seen, start_time)
            progress.set_postfix(
                loss="{:.3f}".format(loss_meter.avg),
                lr="{:.2e}".format(_current_lr(optimizer)),
                ips="{:.1f}".format(imgs_per_sec),
                mem=_cuda_memory_summary(),
                refresh=False,
            )

            if iteration % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Img/s: {:.1f}, "
                    "GPU peak GB: {}".format(
                        epoch,
                        iteration,
                        len(train_loader),
                        loss_meter.avg,
                        imgs_per_sec,
                        _cuda_memory_summary(),
                    )
                )

        scheduler.step(epoch)
        logger.info(
            "Epoch {} done. Time: {:.1f}s, Img/s: {:.1f}, GPU peak GB: {}".format(
                epoch,
                time.time() - start_time,
                _samples_per_second(samples_seen, start_time),
                _cuda_memory_summary(),
            )
        )

        if epoch % checkpoint_period == 0:
            checkpoint_path = os.path.join(
                cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_au_stage1_{}.pth'.format(epoch)
            )
            _save_model_state(model, checkpoint_path)
            logger.info("Saved Stage 1 checkpoint: {}".format(checkpoint_path))

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
    
    model = _prepare_cuda_model(model, logger, "Stage 2", train_loader.batch_size)
    text_model = _unwrap_data_parallel(model)

    loss_meter = AverageMeter()
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    final_results = None
    best_disfa8_f1 = -1.0
    history_records = []
    disable_itc = False
    itc_loss_weight = 0.1
    
    all_start_time = time.time()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        _reset_cuda_peak_memory()
        samples_seen = 0
        loss_meter.reset()

        model.train()
        epoch_train_lr = _current_lr(optimizer)
        logger.info("Stage 2 LR: {:.2e}".format(epoch_train_lr))
        progress = tqdm(
            train_loader,
            desc="Stage2 E{}/{}".format(epoch, epochs),
            total=len(train_loader),
            dynamic_ncols=True,
            leave=False,
        )
        for iteration, (img, target, _, _, _) in enumerate(progress, start=1):
            optimizer.zero_grad()
            img = img.to(device)
            target = target.to(device).float()
            _assert_binary_targets(target, "Stage 2", epoch, iteration)
            
            with torch.amp.autocast('cuda', enabled=False):
                # model returns [logits_list], [feat_list], img_feat_proj
                score, _, img_feat_proj = model(img)

            score = _as_float_tensor_list(score)
            score_tensors = score if isinstance(score, list) else [score]
            for index, score_tensor in enumerate(score_tensors):
                _assert_finite_tensor(
                    score_tensor,
                    "classifier logits[{}]".format(index),
                    "Stage 2",
                    epoch,
                    iteration,
                )

            loss_itc_value = None
            with torch.amp.autocast('cuda', enabled=False):
                # Classification and BCE/ITC math stay in fp32. Weighted BCE can
                # overflow in fp16 before GradScaler has a chance to react.
                loss_cls = loss_fn(score, target)
                loss_cls_value = float(loss_cls.detach().item())
                loss = loss_cls

                if itc_loss_weight > 0.0 and not disable_itc:
                    # Optional: Maintain Image-Text Alignment. This branch is
                    # frozen in Stage 2, so keep it in fp32 and out of the
                    # autograd graph. If an old/unstable Stage 1 prompt makes
                    # ITC non-finite, continue with the main AU BCE objective.
                    with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                        text_features = text_model(get_text=True)

                    img_feat_proj_itc = F.normalize(
                        img_feat_proj.float(), dim=-1, eps=1e-6
                    )
                    text_features_itc = F.normalize(
                        text_features.float(), dim=-1, eps=1e-6
                    )
                    logits_itc = (img_feat_proj_itc @ text_features_itc.t()) / 0.07

                    if _is_finite_tensor(logits_itc):
                        loss_itc = F.binary_cross_entropy_with_logits(
                            logits_itc, target
                        )
                        loss_itc_value = float(loss_itc.detach().item())
                        loss = loss_cls + itc_loss_weight * loss_itc
                    else:
                        disable_itc = True
                        logger.warning(
                            "Disabling Stage 2 ITC regularization after non-finite "
                            "ITC logits at epoch {}, iteration {}. {}. "
                            "Continuing with Weighted BCE only; regenerate the "
                            "Stage 1 checkpoint if you need ITC regularization."
                            .format(
                                epoch,
                                iteration,
                                _tensor_debug_summary(logits_itc),
                            )
                        )
            _assert_finite_loss(loss, "Stage 2", epoch, iteration)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])
            samples_seen += img.shape[0]
            imgs_per_sec = _samples_per_second(samples_seen, start_time)
            best_text = (
                "{:.4f}".format(best_disfa8_f1) if best_disfa8_f1 >= 0 else "n/a"
            )
            itc_text = "off"
            if not disable_itc:
                itc_text = (
                    "{:.3f}".format(loss_itc_value)
                    if loss_itc_value is not None
                    else "pending"
                )
            progress.set_postfix(
                loss="{:.3f}".format(loss_meter.avg),
                cls="{:.3f}".format(loss_cls_value),
                itc=itc_text,
                lr="{:.2e}".format(_current_lr(optimizer)),
                best=best_text,
                ips="{:.1f}".format(imgs_per_sec),
                mem=_cuda_memory_summary(),
                refresh=False,
            )

            if iteration % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, BCE: {:.3f}, "
                    "ITC: {}, Lr: {:.2e}, Best DISFA-8 F1: {}, Img/s: {:.1f}, "
                    "GPU peak GB: {}"
                    .format(
                        epoch,
                        iteration,
                        len(train_loader),
                        loss_meter.avg,
                        loss_cls_value,
                        itc_text,
                        _current_lr(optimizer),
                        best_text,
                        imgs_per_sec,
                        _cuda_memory_summary(),
                    )
                )

        scheduler.step()
        logger.info(
            "Epoch {} done. Time: {:.1f}s, Img/s: {:.1f}, GPU peak GB: {}".format(
                epoch,
                time.time() - start_time,
                _samples_per_second(samples_seen, start_time),
                _cuda_memory_summary(),
            )
        )

        if epoch % checkpoint_period == 0:
            checkpoint_path = os.path.join(
                cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_au_stage2_{}.pth'.format(epoch)
            )
            _save_model_state(model, checkpoint_path)
            logger.info("Saved Stage 2 checkpoint: {}".format(checkpoint_path))

        if epoch % eval_period == 0 or epoch == epochs:
            final_results, best_disfa8_f1 = _run_stage2_eval(
                cfg=cfg,
                model=model,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                train_loss=loss_meter.avg,
                lr=epoch_train_lr,
                best_metric=best_disfa8_f1,
                history_records=history_records,
                itc_enabled=not disable_itc,
                logger=logger,
            )

    if final_results is None:
        final_results, best_disfa8_f1 = _run_stage2_eval(
            cfg=cfg,
            model=model,
            val_loader=val_loader,
            device=device,
            epoch=epochs,
            train_loss=loss_meter.avg,
            lr=_current_lr(optimizer),
            best_metric=best_disfa8_f1,
            history_records=history_records,
            itc_enabled=not disable_itc,
            logger=logger,
        )

    logger.info("Total training time: {:.1f}s".format(time.time() - all_start_time))
    return final_results
