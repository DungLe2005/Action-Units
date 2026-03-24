### CLIP-ReID to Action Unit Detection

Implementation Specification using DISFA Dataset

### 1. Objective

This project extends the architecture of
CLIP-ReID
to perform Facial Action Unit (AU) detection.

The system detects facial muscle movements and generates a natural language description of the expression.

Final pipeline:

Face Image
↓
CLIP Encoder
↓
AU Vector
↓
Natural Language Description

Example output:

"The person raises the cheeks and pulls the lip corners upward,
indicating a happy expression."

### 2. Design Principles

To ensure compatibility with the original CLIP-ReID implementation:

KEEP the original repository structure
KEEP the two-stage training strategy
REUSE CLIP visual encoder
REPLACE ReID classifier with AU classifier

Only the following components should be modified:

dataset loader
classification head
loss function
evaluation metrics

### 3. Dataset

Dataset used:

DISFA

DISFA provides frame-level facial action unit annotations with intensity values ranging from 0–5.

Selected Action Units

The system detects 12 facial action units:

AU1
AU2
AU4
AU5
AU6
AU9
AU12
AU15
AU17
AU20
AU25
AU26

Label dimension:

12

### 4. Label Processing

DISFA annotations are stored in AU text files.

Example structure:

Labels/
SN001/
Trial_1/
AU1.txt
AU2.txt
AU4.txt

Each file contains frame intensity annotations.

Example:

000 0
001 2
002 4
003 0

Format:

frame_id intensity
4.1 Intensity to Binary Conversion

AU detection is treated as a multi-label classification problem.

Intensity is converted to binary activation.

Rule:

AU_active = 1 if intensity ≥ 2
AU_active = 0 if intensity < 2

Example:

frame intensity binary

000 0 0
001 1 0
002 2 1
003 4 1
4.2 Label Aggregation

Each frame must contain a complete AU vector.

Example:

Frame: 000.jpg

AU1 = 0
AU2 = 0
AU4 = 1
AU5 = 0
AU6 = 1
AU9 = 0
AU12 = 1
AU15 = 0
AU17 = 0
AU20 = 0
AU25 = 1
AU26 = 0

Final AU vector:

[0,0,1,0,1,0,1,0,0,0,1,0]
4.3 Label Generation Script

Labels are generated using:

prepare_data.py

Processing pipeline:

DISFA AU text files
↓
parse intensity
↓
convert to binary label
↓
merge AU labels per frame
↓
generate labels.csv

Output file:

AUs_DATA/labels.csv

Example:

image_path,AU1,AU2,AU4,AU5,AU6,AU9,AU12,AU15,AU17,AU20,AU25,AU26

SN001/A1/000.jpg,0,0,1,0,1,0,1,0,0,0,1,0
SN001/A1/001.jpg,0,0,0,0,0,0,1,0,0,0,1,0

Each row contains:

image_path

- 12 AU labels

### 5. Dataset Loader Modification

File to modify:

datasets/make_dataloader.py

Original CLIP-ReID output:

image, person_id

New output:

image, au_vector

Pseudo code:

def **getitem**(self, index):

    image = read_image(path)

    au_label = load_from_csv(index)

    return image, au_label

### 6. Image Preprocessing

Face Detection

Recommended detectors:

RetinaFace
MTCNN

Processing pipeline:

detect face
↓
crop bounding box
↓
align eyes
Image Size

All images resized to:

224 × 224
Normalization

Use CLIP normalization:

mean = [0.48145466, 0.4578275, 0.40821073]

std = [0.26862954, 0.26130258, 0.27577711]
Data Augmentation

Recommended augmentations:

RandomHorizontalFlip
RandomRotation (≤ 5°)
ColorJitter
RandomBrightness

Large geometric transformations should be avoided.

### 7. Model Architecture

Backbone reused from CLIP-ReID:

CLIP Image Encoder

Possible backbones:

ViT-B/16
ResNet-50

### 8. BNNeck

CLIP-ReID contains a BNNeck layer.

Structure:

Feature
↓
BatchNorm1D
↓
Normalized Feature

BNNeck must be preserved.

### 9. AU Classification Head

Replace the ReID classifier.

Original classifier:

Linear(feature_dim, num_identity)

New AU classifier:

Linear(feature_dim, 12)

Activation:

Sigmoid

### 10. Two-Stage Training Strategy

The training procedure follows the original two-stage strategy proposed in CLIP-ReID.

This strategy stabilizes training by first learning text prompts and then aligning image features with the learned text representations.

Stage 1 — Prompt Learning
Objective

Learn task-specific text prompts while keeping the pretrained CLIP encoders fixed.

This stage adapts the CLIP text space to the target task.

Prompt Structure

CLIP-ReID introduces learnable prompt tokens.

Text template:

"A photo of a [X1][X2]...[XM] person"

Where

[X1][X2]...[XM]

are learnable embedding vectors.

These tokens are optimized during training.

Training Configuration

Frozen components:

CLIP Image Encoder
CLIP Text Encoder

Trainable components:

Prompt Tokens
Classification Head
BNNeck
Training Pipeline
Image
↓
CLIP Image Encoder (frozen)
↓
Image Feature

Prompt Template
↓
CLIP Text Encoder (frozen)
↓
Text Feature

Image Feature × Text Feature
↓
Similarity Matrix
↓
Loss
Loss Functions

Two losses are used:

1. Image-Text Contrastive Loss

Encourages alignment between image and text embeddings.

L_ITC

2. Identity Classification Loss

Encourages identity discrimination.

L_ID

Total loss:

L = L_ITC + L_ID
Training Outcome

At the end of Stage 1:

The prompt tokens encode task-specific semantics

The text embedding space becomes identity-aware

However:

Image encoder remains unchanged
Stage 2 — Image Feature Alignment
Objective

Adapt the image encoder so that image features align with the learned text representations.

Training Configuration

Frozen components:

Prompt Tokens
CLIP Text Encoder

Trainable components:

CLIP Image Encoder
BNNeck
Classification Head
Training Pipeline
Image
↓
CLIP Image Encoder (trainable)
↓
Image Feature

Prompt Template
↓
CLIP Text Encoder (frozen)
↓
Text Feature

Image Feature × Text Feature
↓
Similarity Matrix
↓
Loss
Loss Functions

Same losses as Stage 1:

L_ITC

- L_ID

This stage optimizes the image encoder so that:

image embeddings → align with text embeddings
Training Outcome

At the end of Stage 2:

Image features become task-specific

Image embeddings are aligned with the learned prompt representations

Training Summary

The complete training process can be summarized as:

Stage 1:

Learn task-specific prompt tokens
while keeping CLIP encoders frozen

Stage 2:

Fine-tune the image encoder
to align visual features with learned prompts

This two-stage process stabilizes training and preserves the pretrained knowledge of CLIP.

Adaptation for Action Unit Detection

For Action Unit detection, the same two-stage strategy is preserved.

Text prompts describe facial muscle activations.

Example prompts:

"A face showing AU12 (lip corner puller)"

"A face showing AU6 (cheek raiser)"

"A face showing AU1 (inner brow raiser)"

The training objective becomes aligning facial images with AU descriptions.

### 11. Handling Class Imbalance

DISFA contains strong AU imbalance.

Example:

AU12 frequent
AU9 rare

Solution:

Weighted BCE

Weight formula:

w_i = N_total / (2 × N_positive_i)

### 12. Evaluation Metrics

AU detection is evaluated using:

F1-score per AU
Average F1-score
Accuracy
AUC

Example:

## AU F1

AU1 0.71
AU4 0.65
AU12 0.83

### 13. Natural Language Explanation Module

New file to implement:

au_explainer.py

Purpose:

convert AU vector into natural language
AU Phrase Mapping

Example mapping:

AU1 → raises the inner brows
AU2 → raises the outer brows
AU4 → lowers the brows
AU6 → raises the cheeks
AU12 → pulls the lip corners upward
AU25 → parts the lips
AU26 → drops the jaw
Sentence Template

Example templates:

"The person [action1] while [action2]."

"The face shows [action1] and [action2],
suggesting [emotion]."
Emotion Hint (Optional)

Rule-based emotion inference.

Examples:

AU6 + AU12 → happiness
AU4 + AU15 → sadness
AU1 + AU2 + AU5 → surprise

Example output:

"The person raises the cheeks and pulls the lip corners upward,
indicating a happy expression."

### 14. Files to Modify and Add

The repository structure of CLIP-ReID must remain unchanged.

Only the following files need modification.

Files to Modify
datasets/make_dataloader.py
model/make_model.py
processor/processor.py
configs/\*.yaml

Modifications include:

load AU labels
replace classifier
change loss function
update training parameters
Files to Add
datasets/disfa.py
model/au_head.py
au_explainer.py
prepare_data.py

Purpose:

DISFA dataset loader
AU classifier head
natural language explanation
label generation

### 15. Inference Pipeline

Final inference pipeline:

face_image
↓
CLIP encoder
↓
feature
↓
AU classifier
↓
AU vector
↓
au_explainer
↓
sentence

### 16. Example Output

Input:

face.jpg

Model prediction:

AU Vector

[1,1,0,0,1,0,1,0,0,0,1,0]

Generated sentence:

"The person raises both inner and outer brows while pulling
the lip corners upward and slightly parting the lips,
suggesting a joyful expression."

### 17. Conclusion

This project extends CLIP-ReID into a facial action unit detection system with explainable outputs.

Key components:

CLIP visual encoder
BNNeck representation
two-stage training
multi-label AU classification
natural language explanation

The resulting architecture enables both accurate AU detection and interpretable facial expression descriptions.
