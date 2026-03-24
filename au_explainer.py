class AUExplainer:
    def __init__(self):
        # Mapping AU index (0-11) to its descriptive phrase
        # AU_LIST = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
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
        
        # Rule-based emotion inference
        # keys are set of active AU indices
        self.emotion_rules = [
            ({4, 6}, "happy"),          # AU6 + AU12
            ({2, 7}, "sad"),            # AU4 + AU15
            ({0, 1, 3}, "surprised"),   # AU1 + AU2 + AU5
            ({2, 5, 8}, "angry"),       # AU4 + AU9 + AU17
            ({3, 9}, "fearful"),        # AU5 + AU20
            ({5}, "disgusted")          # AU9
        ]

    def infer_emotion(self, active_aus):
        active_set = set(active_aus)
        for required_aus, emotion in self.emotion_rules:
            if required_aus.issubset(active_set):
                return emotion
        return "neutral"

    def explain(self, au_vector):
        """
        au_vector: list or array of 12 binary values (0 or 1)
        Returns: String description
        """
        active_indices = [i for i, val in enumerate(au_vector) if val == 1]
        
        if not active_indices:
            return "The person has a neutral facial expression."

        phrases = [self.au_mapping[i] for i in active_indices]
        emotion = self.infer_emotion(active_indices)

        # Build sentence
        if len(phrases) == 1:
            desc = f"The person {phrases[0]}"
        elif len(phrases) == 2:
            desc = f"The person {phrases[0]} and {phrases[1]}"
        else:
            desc = f"The person {', '.join(phrases[:-1])}, and {phrases[-1]}"

        if emotion != "neutral":
            desc += f", indicating a {emotion} expression."
        else:
            desc += "."

        return desc

# Example usage:
if __name__ == "__main__":
    explainer = AUExplainer()
    # Mock happy AU vector: AU6 (idx 4) and AU12 (idx 6) are 1
    happy_vec = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    print(explainer.explain(happy_vec))
    
    # Mock angry
    angry_vec = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    print(explainer.explain(angry_vec))
