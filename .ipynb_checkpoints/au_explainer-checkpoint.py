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
            0: "nâng lông mày trong lên",             # AU1
            1: "nâng lông mày ngoài lên",             # AU2
            2: "hạ lông mày xuống",                   # AU4
            3: "mở rộng mắt",                    # AU5
            4: "nâng má lên",                  # AU6
            5: "nhăn mũi",                  # AU9
            6: "kéo khóe môi lên",       # AU12
            7: "hạ khóe môi xuống",          # AU15
            8: "nâng cằm lên",                    # AU17
            9: "kéo căng môi",                 # AU20
            10: "mở môi",                    # AU25
            11: "hạ hàm xuống"                      # AU26
        }

        self.au_mapping_en = {
            0: "raising inner eyebrows",
            1: "raising outer eyebrows",
            2: "lowering eyebrows",
            3: "widening eyes",
            4: "raising cheeks",
            5: "wrinkling nose",
            6: "pulling lip corners up",
            7: "depressing lip corners",
            8: "raising chin",
            9: "stretching lips",
            10: "parting lips",
            11: "dropping jaw"
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
            desc_vn = "Người này có biểu cảm khuôn mặt bình thường."
            return desc_vn, "neutral"

        phrases_vn = [self.au_mapping[i] for i in active_indices]
        phrases_en = [self.au_mapping_en[i] for i in active_indices]

        # Build sentence in Vietnamese
        if len(phrases_vn) == 1:
            desc_vn = f"Người này đang {phrases_vn[0]}."
        elif len(phrases_vn) == 2:
            desc_vn = f"Người này đang {phrases_vn[0]} và {phrases_vn[1]}."
        else:
            desc_vn = f"Người này đang {', '.join(phrases_vn[:-1])}, và {phrases_vn[-1]}."

        # Build sentence in English for CLIP
        if len(phrases_en) == 1:
            desc_en = f"The person is {phrases_en[0]}."
        elif len(phrases_en) == 2:
            desc_en = f"The person is {phrases_en[0]} and {phrases_en[1]}."
        else:
            desc_en = f"The person is {', '.join(phrases_en[:-1])}, and {phrases_en[-1]}."

        # Predict emotion purely from the English descriptive sentence using CLIP
        emotion = self.emotion_predictor.predict(desc_en)

        return desc_vn, emotion


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
