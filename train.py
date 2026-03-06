import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from config import load_config
from utils import set_seed, compute_metrics, EarlyStopping
from dataset import create_dataloaders
from models import CLIPActionUnitDetector

def train(config_path="config.yaml"):
    params = load_config(config_path)
    set_seed(params['train']['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data Loaders
    train_loader, val_loader, _, num_classes = create_dataloaders(params)
    
    # Update config num_classes
    params['data']['num_classes'] = num_classes
    
    # 2. Model setup
    model = CLIPActionUnitDetector(
        model_name=params['model']['backbone'],
        num_classes=num_classes,
        freeze_backbone=params['model']['freeze_backbone'],
        hidden_dim=params['model']['hidden_dim'],
        dropout_rate=params['model']['dropout']
    ).to(device)
    
    # 3. Loss, Optimizer and Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=params['train']['learning_rate'], 
        weight_decay=params['train']['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params['train']['epochs']
    )
    
    # 4. Utilities
    scaler = GradScaler() if params['train']['mixed_precision'] else None
    writer = SummaryWriter(log_dir=params['train']['log_dir'])
    os.makedirs(params['train']['save_dir'], exist_ok=True)
    best_model_path = os.path.join(params['train']['save_dir'], 'best_clip_au.pth')
    
    early_stopping = EarlyStopping(
        patience=params['train']['early_stopping_patience'], 
        verbose=True, 
        path=best_model_path
    )
    
    print("Starting training...")
    for epoch in range(params['train']['epochs']):
        # --- Training phase ---
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['train']['epochs']} [Train]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            
            # Store predictions for train metrics
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            train_preds.append(probs)
            train_labels.append(labels.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())
            
        train_loss /= len(train_loader)
        train_preds = np.vstack(train_preds)
        train_labels = np.vstack(train_labels)
        train_metrics = compute_metrics(train_labels, train_preds, params['eval']['threshold'])
        
        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{params['train']['epochs']} [Val]")
            for images, labels in loop:
                images, labels = images.to(device), labels.to(device)
                
                if scaler:
                    with autocast():
                        logits = model(images)
                        loss = criterion(logits, labels)
                else:
                    logits = model(images)
                    loss = criterion(logits, labels)
                    
                val_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(probs)
                val_labels.append(labels.cpu().numpy())
                
                loop.set_postfix(val_loss=loss.item())
                
        val_loss /= len(val_loader)
        val_preds = np.vstack(val_preds)
        val_labels = np.vstack(val_labels)
        val_metrics = compute_metrics(val_labels, val_preds, params['eval']['threshold'])
        
        scheduler.step()
        
        # --- Logging ---
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('F1/train_macro', train_metrics['f1_macro'], epoch)
        writer.add_scalar('F1/val_macro', val_metrics['f1_macro'], epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        early_stopping(val_metrics['f1_macro'], model, is_loss=False)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
            
    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    import yaml
    try:
        train()
    except Exception as e:
        print(f"Failed to train: {e}")
