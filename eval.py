import torch
import numpy as np
import pprint
from tqdm import tqdm
import csv
import os

from config import load_config
from utils import compute_metrics
from dataset import create_dataloaders
from models import CLIPActionUnitDetector

def evaluate(config_path="config.yaml"):
    params = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    # 1. Data Loaders (We only need test_loader)
    _, _, test_loader, num_classes = create_dataloaders(params)
    
    # 2. Model setup
    model = CLIPActionUnitDetector(
        model_name=params['model']['backbone'],
        num_classes=num_classes,
        hidden_dim=params['model']['hidden_dim']
    ).to(device)
    
    # 3. Load weights
    best_model_path = f"{params['train']['save_dir']}/best_clip_au.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()
    
    test_preds, test_labels = [], []
    
    print("Running evaluation on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            test_preds.append(probs)
            test_labels.append(labels.numpy())
            
    test_preds = np.vstack(test_preds)
    test_labels = np.vstack(test_labels)
    
    metrics = compute_metrics(test_labels, test_preds, params['eval']['threshold'])
    
    print("\n--- Evaluation Results ---")
    print(f"F1 Macro:      {metrics['f1_macro']:.4f}")
    print(f"F1 Micro:      {metrics['f1_micro']:.4f}")
    print(f"Precision:     {metrics['precision_macro']:.4f}")
    print(f"Recall:        {metrics['recall_macro']:.4f}")
    print(f"ROC AUC Macro: {metrics['roc_auc_macro']:.4f}")
    print("\nPer-class F1 Scores:")
    for i, score in enumerate(metrics['per_class_f1']):
        print(f"AU {i}: {score:.4f}")
        
    os.makedirs(params['train']['save_dir'], exist_ok=True)
    
    # Add timestamp or model name to prevent overwriting if needed, but for now fixed name
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(params['train']['save_dir'], f'evaluation_metrics_{timestamp}.csv')
    
    # Flatten metrics for CSV, turning per_class lists into individual columns
    flat_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            for i, val in enumerate(v):
                flat_metrics[f"{k}_AU_{i}"] = float(val)
        else:
            flat_metrics[k] = float(v)
            
    # Check if file exists to write header or just append (we will create a new one with timestamp here, so write)
    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=flat_metrics.keys())
        writer.writeheader()
        writer.writerow(flat_metrics)
        
    print(f"\nEvaluation metrics saved to {results_path}")
        
    return metrics

if __name__ == "__main__":
    evaluate()
