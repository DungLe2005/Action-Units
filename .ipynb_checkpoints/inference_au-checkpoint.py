import torch
import argparse
import os
import numpy as np
from PIL import Image
from config import cfg_base as cfg
from model.make_model import make_model
from datasets.preprocessing import build_au_val_transforms, MediaPipeFaceMeshExtractor
from au_explainer import AUExplainer

def main():
    parser = argparse.ArgumentParser(description="AU Detection Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to face image")
    parser.add_argument("--weight_path", type=str, required=True, help="Path to trained model .ckpt or .pth file")
    parser.add_argument("--config_file", type=str, default="configs/au/vit_base_au.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference (cuda or cpu)")
    parser.add_argument("--no_relation", action="store_true", help="Nếu đặt, sẽ bỏ qua cơ chế quan hệ AU (AURelationalModeling)")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2], help="Train stage behavior (1: stable/soft, 2: sharp/focused)")
    parser.add_argument("--strict", action="store_true", help="Bật chế độ load weights nghiêm ngặt (strict=True)")
    args = parser.parse_args()

    # 1. Setup Config
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.DATASETS.NAMES = 'disfa' # Force DISFA logic (for AU head)
    cfg.freeze()

    # 2. Setup Model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create model with 12 classes
    model = make_model(cfg, num_class=12, camera_num=1, view_num=1)
    
    # Load weights
    print(f"Loading weights from {args.weight_path}...")
    checkpoint = torch.load(args.weight_path, map_location="cpu")
    
    # Handle Lightning .ckpt format vs standard .pth
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove "model." prefix if it exists (Lightning prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    else:
        state_dict = checkpoint
        
    try:
        model.load_state_dict(state_dict, strict=args.strict)
        print(f"Successfully loaded weights (strict={args.strict})")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Try running without --strict or check your checkpoint.")
        if args.strict: return
    
    # Set stage (Stage 2 uses sharper attention temp=0.1)
    if hasattr(model, 'set_train_stage'):
        print(f"Setting model to Stage {args.stage} behavior...")
        model.set_train_stage(args.stage)
        
    model.to(device)
    model.eval()

    # 3. Setup Preprocessing and Landmark Extractor
    transforms = build_au_val_transforms(cfg)
    landmark_extractor = MediaPipeFaceMeshExtractor()
    
    # Load Image
    img_pil = Image.open(args.image_path).convert('RGB')
    
    # Precompute Landmarks (Required for accurate AU inference)
    print("Extracting facial landmarks...")
    img_np = np.array(img_pil)
    landmarks = landmark_extractor(img_np).unsqueeze(0).to(device) # [1, 9, 21, 3]
    
    # Check if landmarks were actually detected
    if torch.all(landmarks == 0):
        print("WARNING: No facial landmarks detected! Inference results will be inaccurate.")
    else:
        print("Landmarks detected successfully.")

    # Apply visual transforms
    img_tensor = transforms(img_pil).unsqueeze(0).to(device) # [1, 3, H, W]

    # 4. Inference
    print("Running model inference...")
    with torch.no_grad():
        # model returns sigmoid probabilities when names=='disfa' and eval mode
        # Important: pass landmarks!
        # We also want to capture the mask inside forward for debugging
        # Since pta_mask is internal, let's use a temporary attribute or return it
        # For simplicity, I'll just add a print inside make_model.py or captures it if possible
        # Actually, let's just use the probability stats for now and add more log if needed.
        au_probs_tensor = model(img_tensor, landmarks=landmarks, use_relation=not args.no_relation)
        
    au_probs = au_probs_tensor.cpu().numpy()[0]
    
    # --- Debug Stats ---
    print(f"\nDebug Stats:")
    print(f"  Probs Range: [{au_probs.min():.4f}, {au_probs.max():.4f}]")
    print(f"  Mean Prob:   {au_probs.mean():.4f}")
    print(f"  Std Prob:    {au_probs.std():.4f}")
    
    # Check for saturation
    if au_probs.mean() > 0.9 and au_probs.std() < 0.05:
        print("CRITICAL: Model output is SATURATED (All AUs are predicted high).")
        print("Possible reasons: Mismatched weights, uninitialized BN, or collapsed attention.")
    
    # Use thresholds if possible (default to 0.5)
    # Note: In a real scenario, you'd use the best thresholds found during validation
    thresholds = np.full(12, 0.5)
    au_vector = (au_probs > thresholds).astype(int)
    
    # 5. Output Results
    explainer = AUExplainer()
    explanation, emotion = explainer.explain(au_vector)
    
    print("\n" + "="*50)
    print(f"Inference Results for: {args.image_path}")
    print("="*50)
    
    AU_NAMES = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
    print("Detected Action Units:")
    active_count = 0
    for i, prob in enumerate(au_probs):
        status = "ON" if prob > 0.999 else "OFF"
        if status == "ON":
            print(f"  - AU{AU_NAMES[i]:<2}: {prob:.4f} (Active)")
            active_count += 1
    
    if active_count == 0:
        print("  - No active AUs detected.")
        
    print("\nGenerated Description:")
    print(f"  {explanation}")
    print(f"\nPredicted Emotion:")
    print(f"  {emotion.capitalize()}")
    print("="*50)

if __name__ == "__main__":
    main()
