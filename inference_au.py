import torch
import argparse
import os
from PIL import Image
from config import cfg_base as cfg
from model.make_model import make_model
from datasets.preprocessing import build_au_val_transforms
from au_explainer import AUExplainer

def main():
    parser = argparse.ArgumentParser(description="AU Detection Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to face image")
    parser.add_argument("--weight_path", type=str, required=True, help="Path to trained model .pth file")
    parser.add_argument("--config_file", type=str, default="configs/au/vit_base_au.yaml", help="Path to config file")
    args = parser.parse_args()

    # 1. Setup Config
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.DATASETS.NAMES = 'disfa' # Force DISFA to use AU head
    cfg.freeze()

    # 2. Setup Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create model with 12 classes (Action Units)
    model = make_model(cfg, num_class=12, camera_num=1, view_num=1)
    
    # Load weights
    print(f"Loading weights from {args.weight_path}...")
    state_dict = torch.load(args.weight_path, map_location=device)
    # Handle DataParallel if needed
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 3. Load and Preprocess Image
    transforms = build_au_val_transforms(cfg)
    img = Image.open(args.image_path).convert('RGB')
    img_tensor = transforms(img).unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        # model returns sigmoid probabilities when names=='disfa' and eval mode
        au_probs = model(img_tensor)
        
    au_probs = au_probs.cpu().numpy()[0]
    
    # Threshold at 0.5 for activation
    au_vector = (au_probs > 0.5).astype(int)
    
    # 5. Explain
    explainer = AUExplainer()
    explanation, emotion = explainer.explain(au_vector)
    
    print("\n" + "="*50)
    print(f"Inference Results for: {args.image_path}")
    print("="*50)
    
    AU_NAMES = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
    print("Detected Action Units:")
    active_aus = []
    for i, prob in enumerate(au_probs):
        if prob > 0.5:
            print(f"  - AU{AU_NAMES[i]:<2}: {prob:.4f}")
            active_aus.append(AU_NAMES[i])
    
    if not active_aus:
        print("  - None")
        
    print("\nGenerated Description:")
    print(f"  {explanation}")
    print(f"\nPredicted Emotion (from Description):")
    print(f"  {emotion.capitalize()}")
    print("="*50)

if __name__ == "__main__":
    main()
