# DISFA SOTA Benchmark

Last updated: 2026-04-30

## Dataset And Protocol

**DISFA** (*The Denver Intensity of Spontaneous Facial Action Database*) is a spontaneous facial action database used heavily for facial Action Unit (AU) analysis.

- Original dataset: 27 videos, about 130k frames, 12 female and 15 male subjects.
- Annotation: frame-level FACS AU intensity on a 0-5 ordinal scale.
- 12 intensity AUs: AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26.
- Common AU detection protocol: binarize intensity `>= 2` as active, evaluate 8 AUs: AU1, AU2, AU4, AU6, AU9, AU12, AU25, AU26.
- Common split/metric: subject-exclusive 3-fold cross-validation, F1-frame / Average F1. Higher is better.
- AU intensity estimation uses the 12 AU intensity labels and reports ICC, MSE, and MAE. ICC is higher-is-better; MSE/MAE are lower-is-better.

## AU Detection SOTA On DISFA

| Rank | Model              |                     Year / venue |   DISFA Avg F1 | Main idea / note                                                                                 |
| ---: | ------------------ | -------------------------------: | -------------: | ------------------------------------------------------------------------------------------------ |
|    1 | **Norface**  |                        ECCV 2024 | **72.7** | Identity normalization + unified FEA framework; strongest reported DISFA detection result found. |
|    2 | **FMAE-IAT** |                          FG 2025 | **70.1** | Facial masked autoencoder pretraining on Face9M + identity adversarial training.                 |
|    3 | **ESCDA**    |         Pattern Recognition 2026 | **69.6** | Ensemble of local self-attention constraining and global dual-directional attention.             |
|    4 | FMAE               |                          FG 2025 |           68.7 | FMAE without the IAT regularization component.                                                   |
|    5 | MGRR-Net           |                        TIST 2024 |           68.2 | Multi-level graph relational reasoning for AU detection.                                         |
|    6 | AUNCE              | Pattern Recognition / arXiv 2025 |           66.4 | Contrastive feature learning for AU detection.                                                   |
|    7 | AUFormer           |                        ECCV 2024 |           66.4 | Parameter-efficient vision transformer for AU detection.                                         |
|    8 | MDHRM              |                        CVPR 2024 |           66.2 | Multi-scale dynamic and hierarchical relationship modeling.                                      |
|    9 | AUNet              |                             2023 |           66.1 | Uses temporal/pretrained auxiliary information.                                                  |
|   10 | GLEE-Net           |                       TCSVT 2024 |           65.5 | Global-local expression embedding.                                                               |
|   11 | AC2D               |                        IJCV 2025 |           65.4 | Adaptively constrained self-attention + causal sample deconfounding.                             |

## Per-AU Detection Details

These rows are useful when the report needs more than the average F1. All values are F1-frame percentages.

| Model             |            AU1 |            AU2 |            AU4 |            AU6 |            AU9 |           AU12 |           AU25 |           AU26 |            Avg |
| ----------------- | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: |
| **Norface** | **76.4** |           66.1 |           74.2 | **58.5** | **57.2** | **81.7** | **97.6** | **69.6** | **72.7** |
| ESCDA             |           65.3 | **66.7** |           72.1 |           56.4 |           55.1 |           77.6 |           94.9 |           68.6 |           69.6 |
| AUNCE             |           61.8 |           58.9 | **74.9** |           49.7 |           56.2 |           73.5 |           92.1 |           64.2 |           66.4 |
| AC2D              |           57.8 |           59.2 |           70.1 |           50.1 |           54.4 |           75.1 |           90.3 |           66.2 |           65.4 |

## AU Intensity Estimation On DISFA

| Model / setting   |      DISFA ICC |      DISFA MSE |      DISFA MAE | Note                                                                                                                         |
| ----------------- | -------------: | -------------: | -------------: | ---------------------------------------------------------------------------------------------------------------------------- |
| **Norface** | **0.67** | **0.22** | **0.17** | Fully supervised AU intensity estimation; strongest overall intensity result found.                                          |
| TAS               |     about 0.52 |              - |     about 0.36 | Semi-supervised trend-aware supervision; useful for low-label settings, not directly comparable to fully supervised Norface. |

## Recommended Report Wording

Use this wording when describing the benchmark:

> On the DISFA AU detection benchmark, recent methods usually binarize AU intensity labels with threshold `>= 2` and evaluate 8 AUs using frame-level F1 under subject-exclusive 3-fold cross-validation. The strongest reported result found is Norface with 72.7 average F1, followed by FMAE-IAT with 70.1 and ESCDA with 69.6.

For intensity estimation:

> For DISFA AU intensity estimation, Norface reports 0.67 ICC, 0.22 MSE, and 0.17 MAE across the 12 annotated intensity AUs, making it a strong current reference point for fully supervised intensity prediction.

## Notes And Caveats

- Detection and intensity estimation are different tasks; do not mix Avg F1 with ICC/MAE/MSE.
- Some 2025-2026 papers claim SOTA but compare only against older baselines. In the consolidated table above, Norface remains the highest DISFA Avg F1 found.
- ESCDA reports 69.6 Avg F1 and claims SOTA against the methods in its paper, but its comparison table does not include Norface or FMAE-IAT.
- TAS is semi-supervised and optimized for limited annotation settings, so its numbers should be cited separately from fully supervised intensity results.

## Sources

- DISFA original paper: https://mohammadmahoor.com/wp-content/uploads/2017/06/DiSFA_Paper_andAppendix_Final_OneColumn1-1.pdf
- Norface, ECCV 2024: https://arxiv.org/abs/2407.15617
- Norface ECVA PDF: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07224.pdf
- FMAE-IAT, FG 2025: https://arxiv.org/abs/2407.11243
- ESCDA, Pattern Recognition 2026: https://xuehuai.net/publications/pdfs/2025-pr-facialDetection.pdf
- AUNCE, Pattern Recognition / arXiv: https://arxiv.org/pdf/2402.06165
- AC2D, IJCV 2025: https://arxiv.org/abs/2410.01251
- TAS, semi-supervised AU intensity estimation: https://arxiv.org/abs/2503.08078
- DISFA leaderboard reference: https://beta.hyper.ai/en/sota/tasks/facial-action-unit-detection/benchmark/facial-action-unit-detection-on-disfa

# **5 dataset tương tự DISFA nhất** để tham chiếu cho AU detection / AU intensity:

| Dataset                                       | Quy mô                                              | Annotation AU/FACS              | Tính chất                                | Gần DISFA ở điểm nào                                         |
| --------------------------------------------- | ---------------------------------------------------- | ------------------------------- | ------------------------------------------ | ----------------------------------------------------------------- |
| **BP4D-Spontaneous**                    | 41 subjects, ~368k frames                            | FACS occurrence + intensity     | Spontaneous, induced emotion, 2D/3D video  | Benchmark phổ biến nhất cùng DISFA cho AU detection/intensity |
| **BP4D+**                               | 140 subjects, ~1.4M frames                           | FACS occurrence + intensity     | Multimodal: 2D, 3D, thermal, physiological | Bản lớn hơn của BP4D, rất hợp nếu muốn multimodal AU      |
| **GFT / Sayette Group Formation Task**  | 96 subjects, 172,800 frames                          | FACS occurrence + intensity     | Unscripted 3-person social interaction     | Tự nhiên hơn DISFA/BP4D, có interaction thật                 |
| **UNBC-McMaster Shoulder Pain Archive** | 25 subjects, 200 sequences, 48,398 FACS-coded frames | FACS AU intensity + pain scores | Spontaneous pain expression                | Rất tốt cho AU intensity, nhất là pain-related AU             |
| **Aff-Wild2 / ABAW AU subset**          | ~547-564 in-the-wild videos, ~2.7-2.8M frames        | 12 AU occurrence labels         | In-the-wild, audio-visual video            | Hiện đại hơn, dùng nhiều trong challenge AU detection       |

**Gợi ý dùng trong report**

* Nếu muốn  **so trực tiếp với DISFA** : ưu tiên  **BP4D** ,  **BP4D+** ,  **GFT** ,  **UNBC-McMaster** .
* Nếu muốn thêm benchmark  **hiện đại/in-the-wild** : dùng  **Aff-Wild2** .
* Nếu chỉ cần dataset có  **AU intensity thật sự** : chọn  **BP4D, BP4D+, GFT, UNBC-McMaster** . Aff-Wild2 chủ yếu là AU occurrence/detection.

Nguồn:
BP4D/BP4D+: [https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)
BP4D paper: [https://www.sciencedirect.com/science/article/pii/S0262885614001012](https://www.sciencedirect.com/science/article/pii/S0262885614001012)
GFT paper: [https://www.ri.cmu.edu/pub_files/2017/5/fg17_gft.pdf](https://www.ri.cmu.edu/pub_files/2017/5/fg17_gft.pdf)
UNBC-McMaster: [https://www.sciencedirect.com/science/article/pii/S0262885611001363](https://www.sciencedirect.com/science/article/pii/S0262885611001363)
Aff-Wild2 / ABAW: [https://affective-behavior-analysis-in-the-wild.github.io/10th](https://affective-behavior-analysis-in-the-wild.github.io/10th)
