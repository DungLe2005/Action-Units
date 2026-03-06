import os
import random
import torch
import numpy as np
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Compute binary multi-label classification metrics.
    
    Args:
        y_true: Ground truth binary labels (N, num_classes)
        y_pred_probs: Predicted probabilities (N, num_classes)
        threshold: Threshold to convert probabilities to binary predictions
    
    Returns:
        dict: f1_macro, f1_micro, precision_macro, recall_macro, roc_auc_macro, per_class_f1
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    metrics = {
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    try:
        metrics['roc_auc_macro'] = roc_auc_score(y_true, y_pred_probs, average='macro')
    except ValueError:
        metrics['roc_auc_macro'] = float('nan')
        
    metrics['per_class_f1'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_class_precision'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_class_recall'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    return metrics

class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='best_model.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_metric, model, is_loss=False):
        score = -val_metric if is_loss else val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model, is_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model, is_loss)
            self.counter = 0

    def save_checkpoint(self, val_metric, model, is_loss):
        """Saves model when metric improves."""
        if self.verbose:
            if is_loss:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_metric:.6f}).  Saving model ...')
                self.val_loss_min = val_metric
            else:
                self.trace_func(f'Validation metric improved. Saving model ...')
        
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
