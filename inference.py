import torch
from PIL import Image

from config import load_config
from dataset import get_transforms
from models import CLIPActionUnitDetector

class AUInference:
    def __init__(self, config_path="config.yaml"):
        self.params = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine num_classes from config or fallback
        self.num_classes = self.params['data'].get('num_classes', 12)
        
        # Load model
        self.model = CLIPActionUnitDetector(
            model_name=self.params['model']['backbone'],
            num_classes=self.num_classes,
            hidden_dim=self.params['model']['hidden_dim']
        ).to(self.device)
        
        best_model_path = f"{self.params['train']['save_dir']}/best_clip_au.pth"
        
        try:
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=True))
            print(f"Loaded weights from {best_model_path}")
        except FileNotFoundError:
            print(f"Warning: Checkpoint not found at {best_model_path}. Please train the model first.")
            
        self.model.eval()
        self.transform = get_transforms(self.params['data']['image_size'], is_train=False)
        self.threshold = self.params['eval']['threshold']

    def predict(self, image_path: str):
        """
        Predicts AUs for a single image.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            dict: Probabilities and binary predictions for each AU.
        """
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}
            
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
        binary_preds = (probs >= self.threshold).astype(int)
        
        # Format results
        results = {
            "probabilities": probs.tolist(),
            "binary_predictions": binary_preds.tolist()
        }
        
        # Optionally, map to AU names if known
        print(f"Predicted AUs (Threshold {self.threshold}):")
        for i, (prob, pred) in enumerate(zip(probs, binary_preds)):
            print(f"AU {i:02d}: {pred} (Prob: {prob:.4f})")
            
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference for AU Detection")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    infer = AUInference(config_path=args.config)
    results = infer.predict(args.image_path)
