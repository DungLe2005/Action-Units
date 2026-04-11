import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np

class AUEvaluator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []

    def update(self, output, target):
        """
        output: tensor [B, 12] - sigmoid output (probabilities)
        target: tensor [B, 12] - binary labels
        """
        self.preds.append(output.cpu().numpy())
        self.targets.append(target.cpu().numpy())

    def compute(self):
        preds = np.concatenate(self.preds, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        
        # Binary predictions based on threshold
        binary_preds = (preds > self.threshold).astype(int)
        
        au_f1 = f1_score(targets, binary_preds, average=None)
        avg_f1 = np.mean(au_f1)
        
        # AUC score requires probabilities (preds)
        try:
            au_auc = roc_auc_score(targets, preds, average=None)
            avg_auc = np.mean(au_auc)
        except ValueError:
            # Handle case where one class has only one label present in batch
            au_auc = [0.0] * 12
            avg_auc = 0.0
            
        acc = accuracy_score(targets, binary_preds)
        
        return {
            "au_f1": au_f1,
            "avg_f1": avg_f1,
            "au_auc": au_auc,
            "avg_auc": avg_auc,
            "accuracy": acc
        }

    def save_metrics_to_csv(self, results, output_dir, filename="au_metrics_report.csv"):
        import pandas as pd
        AU_NAMES = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
        
        rows = []
        # Per-AU metrics
        for i, au_id in enumerate(AU_NAMES):
            rows.append({
                'Metric Scope': f'AU{au_id}',
                'F1 Score': results['au_f1'][i],
                'AUC': results['au_auc'][i],
                'Accuracy': '-' # Accuracy is usually reported overall for multi-label
            })
            
        # Overall metrics
        rows.append({
            'Metric Scope': 'Average/Overall',
            'F1 Score': results['avg_f1'],
            'AUC': results['avg_auc'],
            'Accuracy': results['accuracy']
        })
        
        df = pd.DataFrame(rows)
        save_path = os.path.join(output_dir, filename)
        df.to_csv(save_path, index=False)
        return save_path

def do_train_au(cfg,
             model,
             train_loader,
             val_loader,
             optimizer,
             scheduler,
             loss_fn,
             local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('Start AU Training')
    
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    evaluator = AUEvaluator()
    scaler = amp.GradScaler()
    
    all_start_time = time.time()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        evaluator.reset()

        model.train()
        for n_iter, (img, target, _, _, _, landmarks) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            target = target.to(device)
            landmarks = landmarks.to(device)
            
            with amp.autocast(enabled=True):
                # model returns [logits_list], [feat_list]
                score, _ = model(img, landmarks=landmarks)
                loss = loss_fn(score, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, scheduler.get_lr()[0]))

        scheduler.step()
        end_time = time.time()
        logger.info("Epoch {} done. Time: {:.1f}s".format(epoch, end_time - start_time))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_au_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            evaluator.reset()
            for n_iter, (img, target, _, _, _, landmarks) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    landmarks = landmarks.to(device)
                    # Inference mode returns sigmoid probabilities
                    probs = model(img, landmarks=landmarks)
                    evaluator.update(probs, target)
            
            results = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("Avg F1: {:.4f}, Avg AUC: {:.4f}, Accuracy: {:.4f}"
                        .format(results['avg_f1'], results['avg_auc'], results['accuracy']))
            # Selective log per AU
            for i, au_name in enumerate([1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]):
                logger.info(f"AU{au_name} - F1: {results['au_f1'][i]:.4f}, AUC: {results['au_auc'][i]:.4f}")
                
            torch.cuda.empty_cache()

    logger.info("Total training time: {:.1f}s".format(time.time() - all_start_time))
    
    # Final evaluation and save summary metrics to CSV
    logger.info("Performing final evaluation and saving summary metrics to CSV...")
    model.eval()
    evaluator.reset()
    for n_iter, (img, target, _, _, _, landmarks) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            landmarks = landmarks.to(device)
            probs = model(img, landmarks=landmarks)
            evaluator.update(probs, target)
    
    results = evaluator.compute()
    csv_path = evaluator.save_metrics_to_csv(results, cfg.OUTPUT_DIR)
    logger.info(f"Summary metrics report saved to {csv_path}")
