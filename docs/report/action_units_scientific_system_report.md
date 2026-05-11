# Báo cáo khoa học hệ thống Action Units hiện tại

Ngày lập: 2026-04-29
Phạm vi đọc: source code trong repo, `docs/scientific_model_diagram.drawio`, `docs/system_architecture.drawio`, log huấn luyện trong `trainning_logs/`.

## 1. Tóm tắt hệ thống

Hệ thống hiện tại là một hướng mở rộng từ CLIP-ReID sang bài toán nhận diện Facial Action Units (AUs). CLIP-ReID khai thác không gian ảnh-ngôn ngữ của CLIP bằng các text token có thể học để hỗ trợ bài toán re-identification [7], trong khi AU/FACS mô tả chuyển động mặt bằng các đơn vị hành động quan sát được [1]. Thay vì phân loại định danh người/xe như ReID gốc, nhánh AU chuyển bài toán thành multi-label classification trên 12 AU của DISFA, một dataset AU intensity tự phát được gán nhãn theo FACS [2]:

`AU_LIST = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]`

Pipeline khoa học trong sơ đồ và source có thể hiểu là:

```text
DISFA raw frames + AU intensity labels
  -> intensity-to-binary label generation
  -> face-safe image preprocessing
  -> CLIP visual encoder + BNNeck
  -> AUHead multi-label logits
  -> sigmoid probabilities
  -> thresholded AU vector
  -> rule-based natural-language explanation
```

Ở cấp huấn luyện, hệ thống có hai giai đoạn:

```text
Stage I: prompt-based image-text AU alignment
Stage II: supervised AU recognition fine-tuning
```

Về mặt khoa học, hệ thống đang ở mức research prototype/adaptation system: ý tưởng kiến trúc rõ ràng, nhiều khối đã được triển khai, evaluator AU đã có, và protocol đánh giá đã chuyển sang 3-fold subject-exclusive cho DISFA. Các hướng còn lại để tiến gần SOTA nằm ở mô hình theo vùng/quan hệ/thời gian và calibration theo từng AU.

## 2. Nền tảng nghiên cứu liên quan

- FACS/AU: Facial Action Coding System mô tả biểu cảm bằng các Action Units quan sát được trên cơ mặt. Tài liệu FACS định nghĩa tiêu chí quan sát/coding cho từng AU và cách AUs xuất hiện theo tổ hợp [1].
- DISFA: DISFA là cơ sở dữ liệu Actions Unit, có 27 subject, video độ phân giải cao, AU intensity được chấm theo thang 0-5 bởi FACS coders [2], [3].
- CLIP: CLIP học không gian ảnh-ngôn ngữ bằng cách dự đoán cặp ảnh-caption, cho phép dùng ngôn ngữ để gọi tên khái niệm thị giác [4].
- ViT: ViT biểu diễn ảnh thành chuỗi patch và xử lý bằng Transformer, là backbone phù hợp với cấu hình `ViT-B-16` của hệ thống [5].
- Prompt learning/CoOp: CoOp học các vector context của prompt trong khi giữ tham số pretrained cố định, rất gần với cơ chế `PromptLearner` trong repo [6].
- CLIP-ReID: CLIP-ReID đề xuất chiến lược hai giai đoạn: học token mô tả trong Stage I, sau đó cố định text side để ràng buộc fine-tune image encoder trong Stage II [7].
- BN/BNNeck: BatchNorm là cơ sở của `bottleneck` [8]; trong ReID, BNNeck được phổ biến bởi strong baseline ReID để tách embedding dùng cho metric và classifier [9].
- AU detection hiện đại: nhiều phương pháp AU detection dùng attention/region relation để khai thác vùng cơ mặt cục bộ; ví dụ ARL học attention và quan hệ vùng AU từ nhãn AU [10]. Hệ thống hiện tại chưa có khối attention/landmark riêng.

## 3. Giai đoạn 0 - Chuẩn bị nhãn DISFA

Source chính: `prepare_data.py`, `datasets/disfa.py`
Sơ đồ liên quan: `system_architecture.drawio` khối `prepare_data.py`, `AUs_DATA/labels.csv`

### Input

- Thư mục ảnh: `AUs_DATA/Images/{subject}/{trial}/{frame}.jpg`
- Thư mục nhãn: `AUs_DATA/Labels/{subject}/{trial}/AU{k}.txt`
- Mỗi file `AU{k}.txt` chứa frame id và intensity.
- AU intensity gốc nằm trên thang rời rạc 0-5 theo DISFA [2].

### Process

`prepare_data.py` duyệt subject/trial, đọc 12 file AU theo danh sách:

```text
[AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26]
```

Với mỗi frame, script gom 12 intensity thành một vector nhãn. Nếu một AU không có entry cho frame đó, code dùng mặc định intensity = 0.

### Threshold

Ngưỡng nhị phân hóa trong source:

```text
theta_label = 2
y_k = 1 nếu intensity_k >= 2
y_k = 0 nếu intensity_k < 2
```

Đây là bước chuyển từ AU intensity estimation sang AU occurrence detection. Về khoa học, bước này đơn giản hóa bài toán nhưng làm mất thông tin ordinal 0-5 vốn có trong DISFA [2]. Nếu mục tiêu sau này là nhận diện cường độ AU, cần giữ label 0-5 và dùng loss ordinal/regression thay vì BCE nhị phân.

### Output

File:

```text
AUs_DATA/labels.csv
```

Mỗi dòng có:

```text
image_path, AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26
```

## 4. Giai đoạn 1 - Dataset loader và chia tập

Source chính: `datasets/disfa.py`, `datasets/make_dataloader.py`

### Input

- `AUs_DATA/labels.csv`
- Ảnh RGB tương ứng trong `AUs_DATA/Images`
- Config dataset:

```yaml
DATASETS:
  NAMES: 'disfa'
  ROOT_DIR: 'AUs_DATA'
```

### Process

`DISFA` đọc CSV bằng pandas, tạo danh sách:

```text
(img_path, au_label_vector, camid=0, viewid=0)
```

`make_au_dataloader(cfg, fold_idx)` tạo split subject-exclusive 3-fold. Subject id được lấy từ phần đầu của `image_path`, ví dụ `SN001/Trial_1/000.jpg -> SN001`. Code sort subject, shuffle deterministic bằng `cfg.SOLVER.SEED`, chia bằng `np.array_split`, sau đó dùng:

```text
val_subjects = folds[fold_idx]
train_subjects = all folds except fold_idx
```

Train loader dùng `shuffle=True`, val loader dùng `shuffle=False`. Train và val dùng hai object `DISFA` riêng biệt, vì vậy train transform và val transform không còn bị ghi đè qua cùng một dataset object.

### Threshold

Không có threshold mới ở giai đoạn này. Threshold nhãn đã được áp dụng ở giai đoạn 0.

### Output

```text
train_loader: batch gồm image tensor + AU label vector
val_loader: batch validation
num_aus = 12
pos_weight: tensor 12 chiều tính từ train split
fold_info: train/val subjects, sample count, fold id
```

### Phân tích hướng nghiên cứu tiếp theo

Điểm cẩn trọng trước đây là random frame split và shared transform object. Hai điểm này đã được sửa ở protocol hiện tại: train/val không overlap subject và dùng hai dataset instances riêng. Vì vậy metric 3-fold subject-exclusive đáng tin hơn metric random frame split cũ, dù vẫn nên đọc kèm mean/std và per-AU breakdown.

## 5. Giai đoạn 2 - Tiền xử lý ảnh

Source chính: `datasets/preprocessing.py`, `inference_au.py`

### Input

- PIL RGB face image.
- Kích thước target từ config: `224 x 224`.

### Process

Train transform được thiết kế "face-safe":

```text
Resize(224, 224)
RandomHorizontalFlip(p=0.5)
RandomRotation(degrees=5)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0)
ToTensor()
Normalize(mean=CLIP_MEAN, std=CLIP_STD)
```

Validation/inference transform:

```text
Resize(224, 224)
ToTensor()
Normalize(mean=CLIP_MEAN, std=CLIP_STD)
```

Trong source hiện tại chưa có face detection, face crop hoặc landmark alignment online/offline.

### Threshold

Không có threshold quyết định. Các tham số augmentation có vai trò như giới hạn biến đổi:

```text
rotation <= 5 degrees
horizontal_flip_probability = 0.5
brightness_delta = 0.2
contrast_delta = 0.2
saturation_delta = 0.1
hue_delta = 0.0
```

### Output

Tensor ảnh chuẩn CLIP:

```text
x in R^{3 x 224 x 224}
```

### Phân tích hướng nghiên cứu tiếp theo

Thiết kế này hợp lý cho AU hơn pipeline ReID gốc vì tránh `RandomErasing`, random crop mạnh hoặc biến dạng hình học lớn. AU/FACS phụ thuộc vào chuyển động cơ nhỏ ở mắt, lông mày, mũi, miệng, cằm [1]; việc che/cắt vùng mặt có thể phá nhãn. Việc normalize theo CLIP cũng phù hợp với backbone pretrained của CLIP [4]. Tuy nhiên, nếu ảnh đầu vào không được căn mặt ổn định, hệ thống thiếu một khối face alignment có thể làm giảm khả năng tổng quát.

## 6. Giai đoạn 3 - Biểu diễn thị giác bằng CLIP/ViT và BNNeck

Source chính: `model/make_model.py`, `model/au_head.py`

### Input

Batch tensor:

```text
x in R^{B x 3 x 224 x 224}
```

Config chính:

```yaml
MODEL:
  NAME: 'ViT-B-16'
  STRIDE_SIZE: [16, 16]
  NECK: 'bnneck'
```

### Process

`make_model()` tạo `build_transformer`. Kiến trúc này kế thừa ý tưởng dùng CLIP visual encoder cho downstream task [4], với backbone ViT-B/16 dựa trên patch-based Transformer [5]. Với `DATASETS.NAMES == 'disfa'`, classifier ReID được thay bằng:

```text
AUHead(in_planes, 12)
AUHead(in_planes_proj, 12)
```

CLIP visual encoder tạo các feature:

```text
image_features_last
image_features
image_features_proj
```

Với ViT-B/16, code lấy class token:

```text
img_feature_last = image_features_last[:, 0]
img_feature = image_features[:, 0]
img_feature_proj = image_features_proj[:, 0]
```

Sau đó qua BNNeck. Batch Normalization là phép chuẩn hóa feature theo batch để hỗ trợ ổn định/tăng tốc huấn luyện mạng sâu [8], còn BNNeck trong ReID thường được dùng để tách feature embedding và classifier space [9]:

```text
feat = BatchNorm1d(img_feature)
feat_proj = BatchNorm1d(img_feature_proj)
```

Trong training, model trả:

```text
score = [classifier(feat), classifier_proj(feat_proj)]
features = [img_feature_last, img_feature, img_feature_proj]
img_feat_proj
```

Trong eval với `num_classes == 12`, model trả xác suất:

```text
p = sigmoid(classifier(feat)) nếu TEST.NECK_FEAT == 'after'
p = sigmoid(classifier(img_feature)) nếu TEST.NECK_FEAT != 'after'
```

### Threshold

Ở giai đoạn model chưa threshold nhị phân; chỉ có sigmoid biến logit thành xác suất:

```text
p_k = sigmoid(z_k)
```

### Output

- Training: AU logits `z in R^{B x 12}` từ hai classifier.
- Inference/eval: AU probabilities `p in [0,1]^{B x 12}`.

### Phân tích hướng nghiên cứu tiếp theo

Đây là một transfer-learning AU detector dựa trên representation của CLIP [4]. Lợi thế là backbone đã có tri thức thị giác-ngôn ngữ lớn. Hạn chế là AU là tín hiệu rất cục bộ theo vùng cơ mặt [1]; model hiện tại chỉ dùng global class token/embedding, chưa có attention head chuyên biệt cho vùng brow/eye/nose/mouth/jaw như các hướng AU detection dùng attention và relation learning [10].

## 7. Giai đoạn 4 - Stage I: Prompt-based AU semantic alignment

Source chính: `processor/processor_au_2stage.py`, `model/make_model.py`, `solver/make_optimizer_prompt.py`
Sơ đồ liên quan: `scientific_model_diagram.drawio` vùng "Stage I: Prompt-Based AU Semantic Alignment"

### Input

- Mini-batch ảnh:

```text
x_i in R^{3 x H x W}
```

- Nhãn AU nhị phân:

```text
y_i in {0,1}^{12}
```

- Prompt template cho DISFA:

```text
"A photo of a face showing X X X X."
```

### Process

Stage I chỉ tối ưu các tham số chứa `"prompt_learner"` trong tên. Image encoder và text encoder về mặt optimizer không được update. Cơ chế này gần với prompt learning trong CoOp, nơi các context vectors được học để thích nghi CLIP cho downstream recognition [6], và cũng tương ứng với Stage I của CLIP-ReID, nơi learnable tokens được dùng để tạo mô tả mơ hồ cho class/ID trong text encoder [7].

Luồng xử lý:

```text
image_features = model(x=img, get_image=True)
text_features = model(get_text=True)
image_features = normalize(image_features)
text_features = normalize(text_features)
S = image_features @ text_features.T / temperature
loss = BCEWithLogitsLoss(S, y)
```

### Threshold

Không có threshold phân loại. Có temperature cho similarity:

```text
temperature = 0.07
S in R^{B x 12} = V T_AU^T / 0.07
```

### Output

- 12 AU text prototypes/learned prompt embeddings.
- Checkpoint Stage I:

```text
{MODEL.NAME}_au_stage1_{epoch}.pth
```

### Phân tích hướng nghiên cứu tiếp theo

Đây là adaptation của tư tưởng CLIP-ReID và CoOp sang AU detection [6], [7]. Thay vì text label rõ như "inner brow raiser", source hiện tại học 12 context embeddings trong một template chung. Vì vậy, chữ "semantic" ở đây nên hiểu là semantic alignment trong latent CLIP space [4], không phải prompt ngôn ngữ tự nhiên có mô tả AU tường minh. Nếu muốn tăng tính giải thích khoa học, có thể khởi tạo hoặc cố định một phần prompt bằng tên AU thật, ví dụ `"a face showing inner brow raiser"` cho AU1.

## 8. Giai đoạn 5 - Stage II: Multi-label AU recognition fine-tuning

Source chính: `processor/processor_au_2stage.py`, `loss/au_loss.py`, `loss/make_loss.py`, `solver/make_optimizer_prompt.py`
Sơ đồ liên quan: `scientific_model_diagram.drawio` vùng "Stage II: Multi-Label Facial Action Unit Recognition"

### Input

- Mini-batch ảnh đã preprocess.
- Nhãn AU nhị phân `y in {0,1}^{B x 12}`.
- Prompt/text prototypes học từ Stage I.
- Model có CLIP visual encoder, BNNeck, AUHead.

### Process

Optimizer Stage II đóng băng phần text side/prompt side, tương ứng tinh thần Stage II của CLIP-ReID: sau khi học text tokens, fine-tune image encoder dưới ràng buộc từ text features [7].

```text
text_encoder
prompt_learner
```

Các phần còn lại có thể train:

```text
image_encoder
bottleneck / bottleneck_proj
classifier / classifier_proj
```

Forward training:

```text
score, _, img_feat_proj = model(img)
loss_cls = WeightedBCELoss(score, target)
text_features = model(get_text=True)
logits_itc = normalize(img_feat_proj) @ normalize(text_features).T / 0.07
loss_itc = BCEWithLogits(logits_itc, target)
loss = loss_cls + 0.1 * loss_itc
```

`WeightedBCELoss` dùng `BCEWithLogitsLoss(pos_weight=...)`. Trong protocol hiện tại, `pos_weight` được tính từ train split của từng fold:

```text
pos_weight_k = num_negative_k / num_positive_k
```

Nếu một AU không có positive sample trong train split, code dùng `pos_weight=1.0` cho AU đó và log warning.

### Threshold

Stage II training không threshold xác suất; BCE nhận logits trực tiếp. Các hằng số quan trọng:

```text
temperature_ITC = 0.07
lambda_ITC = 0.1
```

### Output

- Checkpoint Stage II:

```text
{MODEL.NAME}_au_stage2_{epoch}.pth
```

- Model có thể xuất xác suất AU:

```text
p in [0,1]^{12}
```

### Phân tích hướng nghiên cứu tiếp theo

Đây là multi-label supervised fine-tuning. Weighted BCE phù hợp vì nhiều AU hiếm hơn các AU khác trong DISFA, một dataset có phân bố AU phụ thuộc subject và chuỗi video [2]. Protocol hiện tại đã tính `pos_weight` từ train split của từng fold, nên trọng số lớp bám sát phân bố dữ liệu đánh giá. Tuy vậy, BCE độc lập theo từng AU không mô hình hóa trực tiếp quan hệ đồng xuất hiện giữa AUs; phần shared backbone và loss ITC có thể học gián tiếp, nhưng chưa có graph/attention relation module chuyên biệt như các hướng AU relation learning [10].

## 9. Giai đoạn 6 - Đánh giá

Source chính: `processor/processor_au_2stage.py`, log trong `trainning_logs/`
Sơ đồ liên quan: `system_architecture.drawio` khối "F1 / AUC / accuracy"

### Input

- Xác suất dự đoán:

```text
p in [0,1]^{B x 12}
```

- Ground-truth:

```text
y in {0,1}^{B x 12}
```

### Process

Thiết kế trong processor gọi:

```python
evaluator = AUEvaluator()
evaluator.update(probs, target)
results = evaluator.compute()
```

Metric dự kiến:

```text
avg_f1 / macro-F1
avg_auc / macro-AUROC
accuracy
per-AU F1
```

DISFA paper cũng báo cáo các metric cho AU presence/absence và AU intensity; trong bối cảnh occurrence detection, F1/Kappa thường phù hợp hơn accuracy thuần vì dữ liệu AU mất cân bằng [2].

### Threshold

Trong source inference là:

```text
theta_pred = 0.5
hat_y_k = 1 nếu p_k > 0.5
```

Evaluator hiện tại đã có trong `processor/processor_au.py` và dùng cùng quy tắc với inference: `hat_y_k = 1[p_k > 0.5]`. Vì vậy metric training/validation là fixed-threshold metric, không phải threshold-calibrated F1.

### Output

Log CSV hiện có:

```text
trainning_logs/Đánh giá kết quả training model AUs lần 1.csv
trainning_logs/Training lần 2 (sửa lại quá trình tiền xử lí).csv
```

Hai dòng metric trong log cho thấy lần training thứ hai, sau khi sửa tiền xử lý, đang là kết quả tốt nhất hiện tại:

| Nhóm | Chỉ số | Lần 1 | Lần 2 | Chênh lệch |
| --- | --- | ---: | ---: | ---: |
| Tổng hợp | f1_macro | 0.9164 | 0.9758 | +0.0593 |
| Tổng hợp | f1_micro | 0.9190 | 0.9753 | +0.0563 |
| Tổng hợp | precision_macro | 0.9378 | 0.9832 | +0.0454 |
| Tổng hợp | recall_macro | 0.8976 | 0.9685 | +0.0709 |
| Tổng hợp | roc_auc_macro | 0.9929 | 0.9991 | +0.0063 |
| Per-AU F1 | AU1 | 0.9116 | 0.9736 | +0.0620 |
| Per-AU F1 | AU2 | 0.9353 | 0.9738 | +0.0385 |
| Per-AU F1 | AU4 | 0.9224 | 0.9767 | +0.0543 |
| Per-AU F1 | AU5 | 0.9139 | 0.9706 | +0.0567 |
| Per-AU F1 | AU6 | 0.9398 | 0.9764 | +0.0366 |
| Per-AU F1 | AU9 | 0.8952 | 0.9777 | +0.0824 |
| Per-AU F1 | AU12 | 0.8837 | 0.9653 | +0.0816 |
| Per-AU F1 | AU15 | 0.9013 | 0.9833 | +0.0820 |
| Per-AU F1 | AU17 | 0.9209 | 0.9761 | +0.0552 |
| Per-AU F1 | AU20 | 0.9059 | 0.9780 | +0.0720 |
| Per-AU F1 | AU25 | 0.9380 | 0.9849 | +0.0469 |
| Per-AU F1 | AU26 | 0.9293 | 0.9731 | +0.0438 |

### Phân tích hướng nghiên cứu tiếp theo

Các metric log cũ vẫn cần đọc với cảnh báo vì nhiều khả năng đến từ random frame split. Protocol hiện tại đã bổ sung `python train_au_2stage.py --all_folds`, chạy 3 fold subject-exclusive và xuất `fold_metrics.csv` + `fold_metrics.json`. Primary metric nên đọc là `DISFA-8 Average F1` mean/std qua 3 fold, kèm per-AU F1.

Rủi ro import `processor.processor_au.AUEvaluator`, random frame split, shared transform object và `pos_weight` hard-code đã được khắc phục. Các rủi ro nghiên cứu còn lại là threshold chưa calibration theo từng AU, chưa có AU-region attention/relation modeling, và chưa có temporal modeling cho chuỗi video.

## 10. Giai đoạn 7 - Inference trên một ảnh

Source chính: `inference_au.py`

### Input

Command-line input:

```text
--image_path path/to/face.jpg
--weight_path path/to/model.pth
--config_file configs/au/vit_base_au.yaml
```

### Process

Luồng inference:

```text
load cfg
force cfg.DATASETS.NAMES = 'disfa'
build make_model(cfg, num_class=12)
load weights
build_au_val_transforms(cfg)
PIL image -> RGB -> tensor -> batch dimension
model.eval()
au_probs = model(img_tensor)
au_vector = (au_probs > 0.5)
AUExplainer.explain(au_vector)
```

### Threshold

Ngưỡng inference:

```text
theta_inference = 0.5
hat_y_k = 1 nếu p_k > 0.5
```

### Output

- Danh sách AU active kèm probability.
- Vector nhị phân 12 chiều.
- Câu mô tả tự nhiên.

Ví dụ dạng output:

```text
AU6: 0.91
AU12: 0.88
Description:
The person raises the cheeks and pulls the lip corners upward, indicating a happy expression.
```

### Phân tích hướng nghiên cứu tiếp theo

Inference hiện tại là deterministic thresholding. Nếu muốn đưa vào ứng dụng nghiêm túc, nên hiệu chỉnh threshold theo AU trên validation set vì mỗi AU có base rate khác nhau trong DISFA [2]. Một threshold 0.5 chung là baseline đơn giản, không nhất thiết tối ưu F1 hoặc precision/recall.

## 11. Giai đoạn 8 - Rule-based semantic verbalization

Source chính: `au_explainer.py`
Sơ đồ liên quan: `scientific_model_diagram.drawio` khối "Rule-constrained semantic verbalization"

### Input

Vector nhị phân:

```text
hat_y in {0,1}^{12}
```

Index mapping trong code:

```text
0 -> AU1
1 -> AU2
2 -> AU4
3 -> AU5
4 -> AU6
5 -> AU9
6 -> AU12
7 -> AU15
8 -> AU17
9 -> AU20
10 -> AU25
11 -> AU26
```

### Process

`AUExplainer` có hai phần. Phần mô tả động tác dựa trên tinh thần FACS/AU: AU là mã cho chuyển động mặt quan sát được, không phải nhãn cảm xúc trực tiếp [1]:

1. Map AU index sang cụm động tác:

```text
AU1 -> raises the inner brows
AU2 -> raises the outer brows
AU4 -> lowers the brows
AU5 -> widens the eyes
AU6 -> raises the cheeks
AU9 -> wrinkles the nose
AU12 -> pulls the lip corners upward
...
```

2. Rule-based emotion hint:

```text
{AU6, AU12} -> happy
{AU4, AU15} -> sad
{AU1, AU2, AU5} -> surprised
{AU4, AU9, AU17} -> angry
{AU5, AU20} -> fearful
{AU9} -> disgusted
```

Rule được kiểm tra theo quan hệ subset: nếu các AU yêu cầu nằm trong active set thì gán emotion đầu tiên match.

### Threshold

Không có threshold trong explainer. Explainer chỉ nhận vector đã nhị phân hóa từ inference.

### Output

Câu tiếng Anh dạng:

```text
The person raises the cheeks and pulls the lip corners upward, indicating a happy expression.
```

Nếu không có AU active:

```text
The person has a neutral facial expression.
```

### Phân tích hướng nghiên cứu tiếp theo

Đây không phải mô hình NLG học từ dữ liệu, mà là lớp giải thích rule-based. Ưu điểm là minh bạch, dễ kiểm soát, nhất quán với tinh thần FACS vì mô tả trực tiếp các hành động cơ mặt [1]. Hạn chế là suy luận cảm xúc chỉ là heuristic: FACS/AU mô tả chuyển động mặt, không chứng minh trực tiếp trạng thái cảm xúc nội tại [1]. Vì vậy nên diễn đạt là "suggesting/indicating" hơn là kết luận chắc chắn.

## 12. Hệ thống hiện tại đang ở đâu về mặt khoa học?

### Điểm mạnh

- Kiến trúc đã chuyển đúng từ single-label ReID sang multi-label AU occurrence detection theo tinh thần FACS/DISFA [1], [2].
- Dùng CLIP/ViT pretrained nên có lợi thế representation so với training từ đầu [4], [5].
- Giữ ý tưởng hai giai đoạn của CLIP-ReID: Stage I học prompt/prototype, Stage II fine-tune image encoder dưới ràng buộc image-text [7].
- Tiền xử lý đã có ý thức bảo toàn tín hiệu mặt, tránh augmentation quá mạnh.
- Có lớp giải thích minh bạch từ AU sang mô tả ngôn ngữ.
- Log hiện có cho thấy metric nội bộ cao, đặc biệt sau lần tiền xử lý thứ hai.

### Điểm chưa đủ chuẩn nghiên cứu

- `processor/processor_au.py` đã được bổ sung, nhưng cần tiếp tục giữ test evaluator trong regression suite khi thay đổi metric/logging.
- Protocol hiện tại đã subject-exclusive 3-fold, nhưng cần chạy đủ fold và báo cáo mean/std trước khi claim benchmark.
- Train/val transform đã tách qua hai dataset instances riêng.
- `pos_weight` đã được tính từ train split từng fold.
- CUDA bị hard-code ở nhiều chỗ (`.cuda()`, `device = "cuda"`), làm giảm reproducibility trên CPU/multi-device.
- Prompt hiện tại chưa chứa tên AU/mô tả AU tường minh, nên semantic alignment chủ yếu là latent learned tokens.
- Model chưa dùng face alignment, landmarks, AU region attention, temporal modeling hoặc relation modeling giữa AUs; đây là các hướng thường xuất hiện trong AU detection hiện đại, ví dụ attention và relation learning [10].
- Threshold inference `0.5` cố định, chưa calibration theo từng AU.

### Định vị khoa học ngắn gọn

Hệ thống hiện tại là một CLIP-based multi-label AU occurrence detector với rule-based verbalization, kết hợp nền tảng CLIP/ViT [4], [5], chiến lược prompt/two-stage của CoOp và CLIP-ReID [6], [7], cùng bài toán AU/FACS trên DISFA [1], [2]. Nó đã có protocol subject-exclusive 3-fold và evaluator phù hợp occurrence detection. Để tiến gần SOTA, hướng tiếp theo là nâng mô hình bằng region-aware attention, relation modeling giữa AUs, temporal modeling theo video, và threshold calibration theo validation fold.

## 13. Phân tích và mô tả theo sơ đồ rút gọn

Mục này diễn giải trực tiếp sơ đồ `scientific_model_diagram_simplified.drawio`. Sơ đồ rút gọn pipeline thành một chuỗi nghiên cứu chính: từ nhãn DISFA dạng intensity, hệ thống tạo nhãn AU occurrence nhị phân, dùng Stage I để học các prompt/prototype AU trong không gian CLIP, dùng Stage II để fine-tune detector AU, sau đó chuyển xác suất thành vector AU và sinh mô tả ngôn ngữ bằng luật.

### Giai đoạn 1 - Dữ liệu DISFA và tạo nhãn nhị phân

**Phân tích:** Đầu vào của hệ thống là các frame khuôn mặt từ DISFA kèm nhãn cường độ AU `a_k in {0,1,2,3,4,5}`. Vì mô hình hiện tại giải bài toán AU occurrence detection, nhãn cường độ được chuyển thành nhãn nhị phân theo công thức `y_k = 1[a_k >= 2]`, tạo vector `y in {0,1}^12` cho 12 AU. Đây là bước làm rõ mục tiêu học: mô hình không dự đoán mức độ mạnh/yếu của AU, mà dự đoán AU có xuất hiện đáng kể hay không.

**Đoạn mô tả:** Ở giai đoạn đầu, dữ liệu DISFA được chuẩn hóa từ dạng intensity annotation sang dạng nhãn xuất hiện AU. Với mỗi frame, 12 giá trị cường độ AU được đọc và nhị phân hóa bằng ngưỡng `2`; các AU có cường độ từ 2 trở lên được xem là xuất hiện, còn các mức 0 hoặc 1 được xem là không xuất hiện rõ. Kết quả của giai đoạn này là một vector nhãn nhị phân 12 chiều, đóng vai trò ground truth thống nhất cho cả quá trình học prompt ở Stage I và fine-tune bộ nhận diện ở Stage II.

### Giai đoạn 2 - Stage I: Prompt-Based Image-Text AU Alignment

**Phân tích:** Stage I dùng CLIP để căn chỉnh đặc trưng ảnh với các embedding prompt đại diện cho 12 AU. Mỗi ảnh được mã hóa thành đặc trưng thị giác `V`, còn mỗi AU có một prompt embedding tương ứng trong ma trận `T_AU`. Độ tương đồng ảnh-văn bản được tính bằng `V T_AU^T / 0.07` và tối ưu bằng `BCEWithLogitsLoss` theo nhãn nhị phân `y`. Mục tiêu khoa học của giai đoạn này là học các prototype AU ở phía ngôn ngữ/semantic space, để mỗi AU có một điểm neo trong không gian CLIP.

**Đoạn mô tả:** Trong Stage I, hệ thống tận dụng khả năng liên kết ảnh-ngôn ngữ của CLIP để học biểu diễn prompt cho từng Action Unit. Ảnh khuôn mặt đi qua visual encoder để tạo vector đặc trưng, trong khi 12 AU được biểu diễn bằng các prompt embedding có thể học. Thay vì huấn luyện ngay một classifier thuần thị giác, hệ thống trước hết tối ưu sự tương đồng giữa ảnh và các prompt AU sao cho những AU xuất hiện trong ảnh có logit tương đồng cao hơn. Đầu ra của giai đoạn này là tập embedding AU đã học, cụ thể là `T_AU` hoặc các tham số `prompt_learner.cls_ctx`.

### Giai đoạn 3 - Chuyển giao prompt AU đã học sang Stage II

**Phân tích:** Kết quả quan trọng nhất của Stage I không phải là vector dự đoán cuối cùng, mà là các AU prompt embeddings đã được tối ưu. Sơ đồ thể hiện bước `freeze / reuse T_AU`, nghĩa là các embedding này được tái sử dụng trong Stage II như một ràng buộc ngữ nghĩa cho quá trình fine-tune detector. Cách thiết kế này giữ tinh thần hai giai đoạn của CLIP-ReID: trước tiên học text/prototype side, sau đó dùng nó để hỗ trợ huấn luyện visual detector.

**Đoạn mô tả:** Sau khi Stage I kết thúc, các prompt AU đã học được giữ lại và chuyển sang Stage II. Chúng đóng vai trò như các prototype ngữ nghĩa của từng AU trong không gian CLIP, giúp giai đoạn fine-tune không chỉ học từ nhãn nhị phân mà còn duy trì liên kết giữa đặc trưng ảnh và biểu diễn AU. Nhờ đó, Stage II có thêm một nguồn regularization từ image-text alignment thay vì chỉ phụ thuộc vào loss phân loại đa nhãn.

### Giai đoạn 4 - Stage II: Fine-tune bộ phát hiện AU occurrence

**Phân tích:** Stage II là giai đoạn huấn luyện bộ nhận diện AU chính. Ảnh được mã hóa bằng CLIP ViT-B/16, hệ thống lấy global class-token features, đưa qua BNNeck và linear `AUHead` để sinh logits `z in R^12`. Loss chính là `Weighted BCEWithLogitsLoss(z, y)` nhằm xử lý bài toán multi-label và mất cân bằng lớp; ngoài ra còn có thành phần phụ `0.1 L_ITC(T_AU)` để giữ liên kết với prompt AU đã học. Sơ đồ cũng ghi rõ phạm vi hiện tại: mô hình dựa vào self-attention của CLIP/ViT trên các patch ảnh, chưa có landmark hoặc AU-region attention block riêng.

**Đoạn mô tả:** Ở Stage II, hệ thống fine-tune một AU occurrence detector dựa trên backbone CLIP ViT-B/16. Đặc trưng ảnh toàn cục từ class token được chuẩn hóa qua BNNeck rồi đưa vào đầu phân loại tuyến tính để tạo 12 logit, mỗi logit tương ứng với một AU. Quá trình huấn luyện dùng Weighted BCE để tăng trọng số cho các AU hiếm, đồng thời bổ sung loss image-text alignment với trọng số `0.1` dựa trên prompt AU từ Stage I. Kết quả của giai đoạn này là một detector có khả năng dự đoán xác suất xuất hiện cho từng AU.

### Giai đoạn 5 - Sinh xác suất AU và quyết định bằng ngưỡng

**Phân tích:** Sau Stage II, đầu ra thô của mô hình là logits `z`. Các logits được chuyển thành xác suất bằng `p = sigmoid(z)`, tạo vector xác suất AU trong khoảng `[0,1]^12`. Ở bước inference, hệ thống dùng ngưỡng cố định `0.5`: `hat_y_k = 1[p_k > 0.5]`. Đây là cách quyết định đơn giản, dễ triển khai và dễ giải thích, nhưng về mặt khoa học có thể chưa tối ưu cho từng AU vì mỗi AU có tần suất xuất hiện khác nhau.

**Đoạn mô tả:** Khi suy luận, mô hình không chỉ trả về nhãn nhị phân mà trước hết sinh xác suất xuất hiện cho từng AU. Mỗi xác suất cho biết mức độ tin cậy của hệ thống rằng AU tương ứng đang xuất hiện trong ảnh. Sau đó, ngưỡng `0.5` được áp dụng để chuyển vector xác suất thành vector AU nhị phân. Vector này là cầu nối giữa detector học sâu và lớp diễn giải ngôn ngữ phía sau.

### Giai đoạn 6 - Diễn giải theo luật và đầu ra cuối cùng

**Phân tích:** Giai đoạn cuối là rule-based verbalization. Hệ thống nhận tập AU đã được threshold, ánh xạ từng AU sang cụm mô tả hành động mặt, rồi bổ sung gợi ý cảm xúc bằng heuristic nếu một tổ hợp AU quen thuộc xuất hiện. Vì lớp này dựa trên luật, nó minh bạch và dễ kiểm soát, nhưng không nên được hiểu là suy luận cảm xúc chắc chắn. Đầu ra cuối cùng gồm ba phần: xác suất AU, vector AU sau threshold, và câu mô tả tự nhiên.

**Đoạn mô tả:** Ở bước cuối, vector AU nhị phân được chuyển thành mô tả ngôn ngữ. Các AU đang hoạt động được ánh xạ thành những cụm mô tả chuyển động mặt như nâng má, kéo khóe môi, hạ lông mày hoặc mở mắt. Nếu tổ hợp AU phù hợp với một mẫu biểu cảm phổ biến, hệ thống thêm một gợi ý cảm xúc ở mức heuristic. Vì vậy, đầu ra cuối cùng vừa giữ thông tin định lượng dưới dạng xác suất AU, vừa cung cấp diễn giải dễ đọc gồm các AU được kích hoạt và mô tả biểu hiện khuôn mặt.

### Đoạn mô tả tổng hợp

Toàn bộ pipeline trong sơ đồ có thể được mô tả như sau: hệ thống bắt đầu từ các frame DISFA cùng nhãn cường độ AU, sau đó chuyển nhãn intensity 0-5 thành nhãn occurrence nhị phân cho 12 AU bằng ngưỡng `2`. Ở Stage I, CLIP được dùng để học các prompt embedding đại diện cho từng AU thông qua loss căn chỉnh ảnh-văn bản. Các prompt embedding này được giữ lại và chuyển sang Stage II, nơi mô hình fine-tune backbone CLIP ViT-B/16, BNNeck và AUHead để dự đoán 12 xác suất AU bằng Weighted BCE kết hợp regularization image-text. Trong suy luận, logits được đưa qua sigmoid và threshold `0.5` để tạo vector AU nhị phân. Cuối cùng, vector này được diễn giải bằng luật thành các cụm mô tả hành động mặt và một gợi ý cảm xúc thận trọng, tạo ra đầu ra gồm xác suất AU, vector AU và mô tả tự nhiên.

## 14. Bảng tổng hợp theo giai đoạn

| Giai đoạn              | Input                     | Process                                    | Threshold / hằng số    | Output                       |
| ------------------------ | ------------------------- | ------------------------------------------ | ------------------------ | ---------------------------- |
| 0. Label generation      | DISFA `AU*.txt`, frames | Parse intensity, merge 12 AUs/frame        | `intensity >= 2`       | `labels.csv`               |
| 1. Dataset loader        | `labels.csv`, images    | `DISFA` subject-exclusive 3-fold       | `cfg.SOLVER.SEED`      | train/val loaders + fold info |
| 2. Preprocessing         | PIL RGB image             | Resize, light augmentation, CLIP normalize | flip 0.5, rotate 5 deg   | tensor `3x224x224`         |
| 3. Visual representation | image tensor              | CLIP ViT-B/16, class token, BNNeck         | sigmoid only at eval     | feature/logits/probs         |
| 4. Stage I alignment     | images, binary AU labels  | image-text similarity, prompt learning     | temperature `0.07`     | learned AU prompts           |
| 5. Stage II recognition  | images, labels, prompts   | Weighted BCE + ITC regularization          | `lambda_ITC = 0.1`     | AU detector checkpoint       |
| 6. Evaluation            | probs, labels             | F1, AUC, accuracy                          | `p > 0.5`              | metrics/logs                 |
| 7. Inference             | image, checkpoint         | preprocess, model, sigmoid                 | `p > 0.5`              | AU vector                    |
| 8. Verbalization         | AU vector                 | phrase mapping + emotion rules             | không có               | natural-language description |

## 15. Hướng nghiên cứu tiếp theo

Sơ đồ roadmap tương ứng được lưu tại `docs/scientific_future_research_roadmap.drawio`, thể hiện các hướng mở rộng từ pipeline hiện tại sang giao thức đánh giá subject-independent, mô hình AU theo vùng, calibration và diễn giải minh bạch hơn.

1. Chạy đủ `python train_au_2stage.py --all_folds` và báo cáo `DISFA-8 Average F1` mean/std.
2. Giữ `processor/processor_au.py` và fold protocol trong regression suite khi thay đổi metric/logging.
3. Báo cáo per-AU F1/AUROC, macro-F1, micro-F1, và mean/std qua folds.
4. Cân nhắc prompt có tên AU thật: `"a face showing AU1 inner brow raiser"`.
5. Cân nhắc threshold riêng theo AU trên validation set.
6. Nếu muốn tiến gần SOTA AU detection, thêm face alignment/landmark hoặc AU-region attention.
7. Sau region attention, bổ sung relation modeling giữa AUs và temporal modeling cho video frames.

## 16. References

[1] P. Ekman, W. V. Friesen, and J. C. Hager, *Facial Action Coding System: The Manual on CD ROM*. Salt Lake City, UT, USA: A Human Face, 2002. [Online]. Available: https://www.paulekman.com/facial-action-coding-system/

[2] S. M. Mavadati, M. H. Mahoor, K. Bartlett, P. Trinh, and J. F. Cohn, "DISFA: A spontaneous facial action intensity database," *IEEE Transactions on Affective Computing*, vol. 4, no. 2, pp. 151-160, Apr.-Jun. 2013, doi: 10.1109/T-AFFC.2013.4.

[3] S. M. Mavadati, M. H. Mahoor, K. Bartlett, P. Trinh, and J. F. Cohn, "DISFA: A spontaneous facial action intensity database," PDF version. [Online]. Available: https://mohammadmahoor.com/wp-content/uploads/2013/08/DiSFA_Paper_andAppendix_Final_OneColumn1.pdf

[4] A. Radford *et al*., "Learning transferable visual models from natural language supervision," in *Proc. 38th International Conference on Machine Learning (ICML)*, 2021, pp. 8748-8763. [Online]. Available: https://proceedings.mlr.press/v139/radford21a.html

[5] A. Dosovitskiy *et al*., "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. International Conference on Learning Representations (ICLR)*, 2021. [Online]. Available: https://arxiv.org/abs/2010.11929

[6] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, "Learning to prompt for vision-language models," *International Journal of Computer Vision*, vol. 130, pp. 2337-2348, 2022. [Online]. Available: https://arxiv.org/abs/2109.01134

[7] S. Li, L. Sun, and Q. Li, "CLIP-ReID: Exploiting vision-language model for image re-identification without concrete text labels," in *Proc. AAAI Conference on Artificial Intelligence*, 2023. [Online]. Available: https://arxiv.org/abs/2211.13977

[8] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. 32nd International Conference on Machine Learning (ICML)*, 2015, pp. 448-456. [Online]. Available: https://proceedings.mlr.press/v37/ioffe15.html

[9] H. Luo, Y. Gu, X. Liao, S. Lai, and W. Jiang, "Bag of tricks and a strong baseline for deep person re-identification," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2019. [Online]. Available: https://arxiv.org/abs/1903.07071

[10] Z. Shao, Z. Liu, J. Cai, and L. Ma, "Facial action unit detection using attention and relation learning," arXiv:1808.03457, 2018. [Online]. Available: https://arxiv.org/abs/1808.03457
