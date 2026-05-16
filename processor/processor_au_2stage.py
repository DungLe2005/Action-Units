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


def _cfg_value(section, name, default):
    return getattr(section, name, default)


def _stage2_amp_enabled(cfg):
    return bool(_cfg_value(cfg.SOLVER.STAGE2, "AMP", False)) and torch.cuda.is_available()


def _lr_group_summary(optimizer):
    grouped = {}
    for group in optimizer.param_groups:
        group_name = group.get("stage2_group", "default")
        grouped.setdefault(group_name, {"count": 0, "lrs": []})
        grouped[group_name]["count"] += 1
        grouped[group_name]["lrs"].append(float(group["lr"]))

    parts = []
    for group_name in sorted(grouped):
        lrs = grouped[group_name]["lrs"]
        if min(lrs) == max(lrs):
            lr_text = "{:.2e}".format(lrs[0])
        else:
            lr_text = "{:.2e}-{:.2e}".format(min(lrs), max(lrs))
        parts.append(
            "{}:{} tensors lr={}".format(
                group_name, grouped[group_name]["count"], lr_text
            )
        )
    return "; ".join(parts)


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


def _score_tensors(values):
    values = _as_float_tensor_list(values)
    return values if isinstance(values, list) else [values]


def _max_abs_tensor_value(tensors):
    max_value = 0.0
    for tensor in tensors:
        current = float(tensor.detach().float().abs().max().item())
        max_value = max(max_value, current)
    return max_value


def _positive_rate_from_logits(tensors):
    if len(tensors) == 1:
        logits = tensors[0].detach().float()
    else:
        logits = torch.stack([tensor.detach().float() for tensor in tensors], dim=0).mean(dim=0)
    return float((torch.sigmoid(logits) > 0.5).float().mean().item())


def _assert_finite_model_parameters(model, stage_name, epoch=None, iteration=None):
    base_model = _unwrap_data_parallel(model)
    for name, param in base_model.named_parameters():
        if not param.is_floating_point():
            continue
        if torch.isfinite(param.detach()).all().item():
            continue
        location = ""
        if epoch is not None and iteration is not None:
            location = " at epoch {}, iteration {}".format(epoch, iteration)
        elif epoch is not None:
            location = " at epoch {}".format(epoch)
        raise FloatingPointError(
            "{} found non-finite parameter {}{}. {}".format(
                stage_name, name, location, _tensor_debug_summary(param)
            )
        )


def _save_stage2_diagnostic(model, cfg, epoch, iteration, reason, logger):
    safe_reason = "".join(
        character if character.isalnum() else "_" for character in str(reason).lower()
    ).strip("_")[:48]
    if not safe_reason:
        safe_reason = "diagnostic"
    checkpoint_path = os.path.join(
        cfg.OUTPUT_DIR,
        "{}_au_stage2_diagnostic_e{}_i{}_{}.pth".format(
            cfg.MODEL.NAME, epoch, iteration, safe_reason
        ),
    )
    _save_model_state(model, checkpoint_path)
    logger.error("Saved Stage 2 diagnostic checkpoint: {}".format(checkpoint_path))
    return checkpoint_path


def _raise_stage2_diagnostic(model, cfg, epoch, iteration, reason, message, logger):
    checkpoint_path = _save_stage2_diagnostic(
        model, cfg, epoch, iteration, reason, logger
    )
    raise FloatingPointError("{} Diagnostic checkpoint: {}".format(message, checkpoint_path))


def _update_stage2_early_stop(
    current_metric,
    best_metric_for_stop,
    epochs_without_improvement,
    min_delta,
):
    if math.isfinite(current_metric) and current_metric > best_metric_for_stop + min_delta:
        return current_metric, 0, True
    return best_metric_for_stop, epochs_without_improvement + 1, False


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
    positive_predictions = 0
    total_predictions = 0
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
            positive_predictions += int((probs > 0.5).sum().item())
            total_predictions += int(probs.numel())
            evaluator.update(probs, target)
    results = evaluator.compute()
    results["eval_positive_rate"] = (
        float(positive_predictions) / float(total_predictions)
        if total_predictions > 0
        else float("nan")
    )
    return results


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
    grad_norm=None,
    max_logit_abs=None,
    train_positive_rate=None,
    stopped_early=False,
    stop_reason="",
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
            grad_norm=grad_norm,
            max_logit_abs=max_logit_abs,
            train_positive_rate=train_positive_rate,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
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
        "Stage 2 diagnostics - GradNorm: {}, MaxLogitAbs: {}, TrainPosRate: {}, "
        "EvalPosRate: {:.4f}".format(
            "n/a" if grad_norm is None else "{:.3f}".format(float(grad_norm)),
            "n/a" if max_logit_abs is None else "{:.3f}".format(float(max_logit_abs)),
            (
                "n/a"
                if train_positive_rate is None
                else "{:.4f}".format(float(train_positive_rate))
            ),
            float(results.get("eval_positive_rate", float("nan"))),
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

            if log_period > 0 and iteration % log_period == 0:
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
    grad_norm_meter = AverageMeter()
    logit_abs_meter = AverageMeter()
    positive_rate_meter = AverageMeter()
    amp_enabled = _stage2_amp_enabled(cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    final_results = None
    best_disfa8_f1 = -1.0
    best_metric_for_early_stop = -1.0
    epochs_without_improvement = 0
    history_records = []
    disable_itc = False
    itc_loss_weight = 0.1
    max_grad_norm = float(_cfg_value(cfg.SOLVER.STAGE2, "MAX_GRAD_NORM", 5.0))
    max_logit_abs_limit = float(_cfg_value(cfg.SOLVER.STAGE2, "MAX_LOGIT_ABS", 0.0))
    early_stop_patience = int(
        _cfg_value(cfg.SOLVER.STAGE2, "EARLY_STOP_PATIENCE", 0)
    )
    early_stop_min_delta = float(
        _cfg_value(cfg.SOLVER.STAGE2, "EARLY_STOP_MIN_DELTA", 0.0)
    )

    logger.info("Stage 2 AMP: {}".format("enabled" if amp_enabled else "disabled"))
    logger.info("Stage 2 LR groups: {}".format(_lr_group_summary(optimizer)))

    all_start_time = time.time()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        _reset_cuda_peak_memory()
        samples_seen = 0
        loss_meter.reset()
        grad_norm_meter.reset()
        logit_abs_meter.reset()
        positive_rate_meter.reset()

        model.train()
        epoch_train_lr = _current_lr(optimizer)
        logger.info(
            "Stage 2 LR: {:.2e} ({})".format(
                epoch_train_lr, _lr_group_summary(optimizer)
            )
        )
        progress = tqdm(
            train_loader,
            desc="Stage2 E{}/{}".format(epoch, epochs),
            total=len(train_loader),
            dynamic_ncols=True,
            leave=False,
        )
        for iteration, (img, target, _, _, _) in enumerate(progress, start=1):
            optimizer.zero_grad(set_to_none=True)
            img = img.to(device)
            target = target.to(device).float()
            _assert_binary_targets(target, "Stage 2", epoch, iteration)

            with torch.amp.autocast('cuda', enabled=amp_enabled):
                # model returns [logits_list], [feat_list], img_feat_proj
                score, _, img_feat_proj = model(img)

            score = _as_float_tensor_list(score)
            score_tensors = _score_tensors(score)
            for index, score_tensor in enumerate(score_tensors):
                if not _is_finite_tensor(score_tensor):
                    _raise_stage2_diagnostic(
                        model,
                        cfg,
                        epoch,
                        iteration,
                        "non_finite_logits",
                        "Stage 2 produced non-finite classifier logits[{}]. {}".format(
                            index, _tensor_debug_summary(score_tensor)
                        ),
                        logger,
                    )
            max_logit_abs = _max_abs_tensor_value(score_tensors)
            if max_logit_abs_limit > 0.0 and max_logit_abs > max_logit_abs_limit:
                _raise_stage2_diagnostic(
                    model,
                    cfg,
                    epoch,
                    iteration,
                    "logit_guardrail",
                    "Stage 2 max logit abs {:.3f} exceeded limit {:.3f}".format(
                        max_logit_abs, max_logit_abs_limit
                    ),
                    logger,
                )
            train_positive_rate = _positive_rate_from_logits(score_tensors)

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
            if not torch.isfinite(loss.detach()).item():
                _raise_stage2_diagnostic(
                    model,
                    cfg,
                    epoch,
                    iteration,
                    "non_finite_loss",
                    "Stage 2 produced a non-finite loss. {}".format(
                        _tensor_debug_summary(loss)
                    ),
                    logger,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            grad_norm_value = float(
                grad_norm.detach().float().item()
                if torch.is_tensor(grad_norm)
                else grad_norm
            )
            if not math.isfinite(grad_norm_value):
                _raise_stage2_diagnostic(
                    model,
                    cfg,
                    epoch,
                    iteration,
                    "non_finite_grad_norm",
                    "Stage 2 produced non-finite gradient norm {}".format(
                        grad_norm_value
                    ),
                    logger,
                )
            scaler.step(optimizer)
            scaler.update()
            _assert_finite_model_parameters(model, "Stage 2", epoch, iteration)

            loss_meter.update(loss.item(), img.shape[0])
            grad_norm_meter.update(grad_norm_value, img.shape[0])
            logit_abs_meter.update(max_logit_abs, img.shape[0])
            positive_rate_meter.update(train_positive_rate, img.shape[0])
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
                grad="{:.2f}".format(grad_norm_value),
                logit="{:.1f}".format(max_logit_abs),
                pos="{:.3f}".format(train_positive_rate),
                lr="{:.2e}".format(_current_lr(optimizer)),
                best=best_text,
                ips="{:.1f}".format(imgs_per_sec),
                mem=_cuda_memory_summary(),
                refresh=False,
            )

            if log_period > 0 and iteration % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, BCE: {:.3f}, "
                    "ITC: {}, GradNorm: {:.3f}, MaxLogitAbs: {:.3f}, "
                    "TrainPosRate: {:.4f}, Lr: {:.2e}, Best DISFA-8 F1: {}, "
                    "Img/s: {:.1f}, GPU peak GB: {}"
                    .format(
                        epoch,
                        iteration,
                        len(train_loader),
                        loss_meter.avg,
                        loss_cls_value,
                        itc_text,
                        grad_norm_value,
                        max_logit_abs,
                        train_positive_rate,
                        _current_lr(optimizer),
                        best_text,
                        imgs_per_sec,
                        _cuda_memory_summary(),
                    )
                )

        scheduler.step()
        _assert_finite_model_parameters(model, "Stage 2", epoch)
        logger.info(
            "Epoch {} done. Time: {:.1f}s, Img/s: {:.1f}, GradNorm: {:.3f}, "
            "MaxLogitAbs: {:.3f}, TrainPosRate: {:.4f}, GPU peak GB: {}".format(
                epoch,
                time.time() - start_time,
                _samples_per_second(samples_seen, start_time),
                grad_norm_meter.avg,
                logit_abs_meter.avg,
                positive_rate_meter.avg,
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
            _assert_finite_model_parameters(model, "Stage 2", epoch)
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
                grad_norm=grad_norm_meter.avg,
                max_logit_abs=logit_abs_meter.avg,
                train_positive_rate=positive_rate_meter.avg,
            )
            current_metric = float(
                final_results.get(PRIMARY_STAGE2_METRIC, float("nan"))
            )
            (
                best_metric_for_early_stop,
                epochs_without_improvement,
                improved_for_early_stop,
            ) = _update_stage2_early_stop(
                current_metric,
                best_metric_for_early_stop,
                epochs_without_improvement,
                early_stop_min_delta,
            )
            if improved_for_early_stop:
                logger.info(
                    "Stage 2 early-stop monitor improved to {:.4f}".format(
                        best_metric_for_early_stop
                    )
                )
            elif early_stop_patience > 0:
                logger.info(
                    "Stage 2 early-stop monitor: no improvement for {}/{} evals".format(
                        epochs_without_improvement, early_stop_patience
                    )
                )
            if (
                early_stop_patience > 0
                and epochs_without_improvement >= early_stop_patience
            ):
                stop_reason = (
                    "{} did not improve by {:.4f} for {} evals".format(
                        PRIMARY_STAGE2_METRIC,
                        early_stop_min_delta,
                        early_stop_patience,
                    )
                )
                if history_records:
                    history_records[-1]["stopped_early"] = True
                    history_records[-1]["stop_reason"] = stop_reason
                    write_stage2_history(cfg.OUTPUT_DIR, history_records)
                logger.info("Early stopping Stage 2: {}".format(stop_reason))
                break

    if final_results is None:
        _assert_finite_model_parameters(model, "Stage 2", epochs)
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
            grad_norm=grad_norm_meter.avg,
            max_logit_abs=logit_abs_meter.avg,
            train_positive_rate=positive_rate_meter.avg,
        )

    logger.info("Total training time: {:.1f}s".format(time.time() - all_start_time))
    return final_results
