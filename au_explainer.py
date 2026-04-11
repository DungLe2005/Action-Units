import torch
import torch.nn.functional as F
from model.clip import clip

class EmotionPredictor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # Load CLIP
        print("Loading CLIP for text-based emotion prediction...")
        self.clip_model, _ = clip.load("ViT-B-16", device=device)
        self.clip_model.eval()
        
        self.emotions = ["happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral"]
        # Prompts designed for zero-shot text matching
        self.emotion_prompts = [f"A face expression that looks {e}" for e in self.emotions]
        
        # Precompute emotion embeddings
        tokens = clip.tokenize(self.emotion_prompts).to(self.device)
        with torch.no_grad():
            self.emotion_features = self.clip_model.encode_text(tokens).float()
            self.emotion_features /= self.emotion_features.norm(dim=-1, keepdim=True)

    def predict(self, description: str):
        tokens = clip.tokenize([description]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        similarity = (text_features @ self.emotion_features.T).squeeze(0)
        best_idx = similarity.argmax().item()
        return self.emotions[best_idx]


class AUExplainer:
    def __init__(self):
        # Mapping AU index (0-11) to its descriptive phrase
        self.au_mapping = {
            0: "raises the inner brows",             # AU1
            1: "raises the outer brows",             # AU2
            2: "lowers the brows",                   # AU4
            3: "widens the eyes",                    # AU5
            4: "raises the cheeks",                  # AU6
            5: "wrinkles the nose",                  # AU9
            6: "pulls the lip corners upward",       # AU12
            7: "depresses the lip corners",          # AU15
            8: "raises the chin",                    # AU17
            9: "stretches the lips",                 # AU20
            10: "parts the lips",                    # AU25
            11: "drops the jaw"                      # AU26
        }
        
        # EmotionPredictor for text-based classification
        self.emotion_predictor = EmotionPredictor()

    def explain(self, au_vector):
        """
        au_vector: list or array of 12 binary values (0 or 1)
        Returns: (String description, Predicted emotion)
        """
        active_indices = [i for i, val in enumerate(au_vector) if val == 1]
        
        if not active_indices:
            desc = "The person has a neutral facial expression."
            return desc, "neutral"

        phrases = [self.au_mapping[i] for i in active_indices]

        # Build sentence
        if len(phrases) == 1:
            desc = f"The person {phrases[0]}."
        elif len(phrases) == 2:
            desc = f"The person {phrases[0]} and {phrases[1]}."
        else:
            desc = f"The person {', '.join(phrases[:-1])}, and {phrases[-1]}."

        # Predict emotion purely from the descriptive sentence using CLIP
        emotion = self.emotion_predictor.predict(desc)

        return desc, emotion


# Example usage:
if __name__ == "__main__":
    explainer = AUExplainer()
    # Mock happy AU vector: AU6 (idx 4) and AU12 (idx 6) are 1
    happy_vec = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    desc, emotion = explainer.explain(happy_vec)
    print(f"Desc: {desc}\nEmotion: {emotion}\n")
    
    # Mock angry
    angry_vec = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    desc, emotion = explainer.explain(angry_vec)
    print(f"Desc: {desc}\nEmotion: {emotion}\n")
