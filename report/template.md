---
title: "Reader disagreement and calibration in chest X-ray classification (VinDr-CXR): progress report"
author: "Mohammad Alaa Al Din Al Hamwi, Abdulrahman Kaddoura, Hamza Rachidi"
date: "6 November 2026"
---

*Milestone 2. All results are preliminary: one internal test split, 12 training epochs, 3 seeds per configuration. Code: <https://github.com/Hamza-69/detection-ml-project>. Live demo: <https://vindr-cxr-aub.fly.dev/>.*

# 1. Problem statement

Each VinDr-CXR [1] training X-ray was annotated by {{DS['readers_per_image'][0]}} of {{DS['n_readers']}} radiologists, who drew boxes for 14 findings and often disagree. We predict, per image, the probability of each finding and of "abnormal" (any finding). **Main question:** does training on *soft labels* (the fraction of readers who marked a finding [2]) give better-calibrated probabilities than *majority* or *union* labels, especially on contested images, without lowering AUROC? **Secondary (model ladder):** how much does a fine-tuned CNN gain over a tuned classical model, per finding, and at what cost? **H1:** soft labels give lower ECE than majority labels, with macro-AUROC at most 0.01 lower. **H2:** on split-vote images, soft-model probabilities track the reader fraction more closely. **H3:** union labels give the highest sensitivity but the worst specificity and calibration. **H4:** the CNN beats the baseline, with a smaller gap on rare findings.

**Changes since the 7 October proposal.** The official test labels need PhysioNet credentialing, which we have not obtained yet, so we test on an **internal split** of the training images (3-reader majority instead of the official 5-reader consensus). Baseline and CNN now share one image-level task and metric set; YOLOv8 detection is a side experiment, at 640 instead of 1024 px. The positive-class weight became an experimental factor, because it changes calibration by itself.

# 2. Data preprocessing

Labels come from the official competition `train.csv` ({{ f"{DS['n_images']:,}" }} images); images from a public 512 px PNG conversion of the DICOMs, which we verified against {{DS['n_verified']}} DICOMs converted ourselves (at most {{DS['max_gray_diff']}} grey level of difference). A reader *marks* finding *k* if they drew at least one box of class *k*; the vote count n ∈ {0,…,3} gives **union** (n ≥ 1), **majority** (n ≥ 2) and **soft** (n/3) targets. Disagreement is the norm: for {{DS['n_findings_over75_split']}} of the 14 findings, over 75% of positive images are split votes (Fig. 1), yet readers never disagree on normal vs. abnormal. Some findings are rare (Pneumothorax: {{int(ST.loc['Pneumothorax','majority_pos'])}} majority-positive images, {{f(ST.loc['Pneumothorax','majority_pos'] / DS['n_images'] * 100, 1)}}%). **Splits** (seed 42, stratified, fixed before training): train {{ f"{DS['split']['train']:,}" }}, validation {{ f"{DS['split']['val']:,}" }}, internal test {{ f"{DS['split']['test']:,}" }}, and a **disagreement set** of {{ f"{DS['split']['disagree']:,}" }} images that keeps all three votes and is never used for training or tuning. The CNN gets 512 px input with light GPU augmentation (rotation ±7°, scale ±10%, shift ±5%, brightness/contrast; no flips, because cardiac findings depend on laterality); the baseline uses 256 px. For detection, the readers' boxes are merged by weighted boxes fusion [3], keeping the {{ f"{DS['n_fused_maj']:,}" }} boxes supported by at least 2 readers.

# 3. Evaluation metrics

All metrics use the internal test set and majority labels unless noted. **Macro-AUROC** over the 14 findings (plus abnormal-vs-normal AUROC), with 95% bootstrap CIs, is threshold-free and does not depend on prevalence, which matters when findings occur on under 1% of images (accuracy would reward always predicting "absent"); the macro average weights rare findings equally. We also compute it **among abnormal images only**, because normal-vs-abnormal separation is easy and inflates per-finding AUROC. **Calibration** is the main question: macro **ECE** (15 bins) [4] checks whether a predicted 30% comes true 30% of the time, and the **Brier score**, a proper scoring rule, combines calibration and discrimination. Both are also reported after temperature scaling [5] and per-class Platt scaling [6], fit on validation. **Sensitivity and specificity at 0.5** test H3. On the **disagreement set** (H2) we compare predictions with the reader fraction k/3: mean |p − k/3| on split votes and Spearman correlation with the vote count. Detection uses AP at IoU 0.4 (the competition metric); cost uses parameters, training time and inference time.

# 4. Classical baseline

Each 256 px image is described by {{ f"{BT['n_features']:,}" }} hand-crafted features: HOG [7] (shape), uniform-LBP histograms [8] on a 4 × 4 grid (texture) and intensity histograms (density). For each target we train an L2 logistic regression on standardised features (C ∈ {10⁻⁴,…,10⁻¹}) and LightGBM [9] (num_leaves ∈ {15, 31}, trees by early stopping) on majority labels, both tuned by 5-fold stratified CV on the training split. CV chose strong regularisation (C = 10⁻³ for {{int((CV[CV.model=='logreg'].best_param == 'C=0.001').sum())}} of 15 targets) and {{CV[CV.model=='lgbm'].best_param.str.extract(r'trees=(\d+)')[0].astype(int).min()}}–{{CV[CV.model=='lgbm'].best_param.str.extract(r'trees=(\d+)')[0].astype(int).max()}} trees.

**Table 1.** Validation: 5-fold CV on train (baselines) or the validation split at the chosen epoch (CNN). Test: baselines with 95% bootstrap CI; CNN mean ± SD over 3 seeds (majority labels, pos_weight).

{{T_BASELINE}}

Validation and test AUROC agree, so the tuning did not overfit; LightGBM is the stronger baseline. Both mostly detect *that* an image is abnormal (abnormal AUROC up to {{f(S.loc['baseline_lgbm','abnormal_auroc'])}}): among abnormal images, macro-AUROC falls to {{f(S.loc['baseline_logreg','macro_auroc_within_abnormal'])}}–{{f(S.loc['baseline_lgbm','macro_auroc_within_abnormal'])}}. They do worst on small or diffuse findings (LightGBM: Nodule/Mass {{f(L.loc['Nodule/Mass','baseline_lgbm'])}}, Calcification {{f(L.loc['Calcification','baseline_lgbm'])}}; Fig. 2).

# 5. Sophisticated model

**Model.** We fine-tune an ImageNet-pretrained DenseNet-121 [10] ({{f(CT['cnn_n_params'] / 1e6, 1)}} M parameters, 15 sigmoid outputs) at 512 px with BCE on the chosen targets, AdamW (lr 10⁻⁴), a cosine schedule and 12 epochs, keeping the epoch with the best validation macro-AUROC. The main setting uses a positive-class weight (#neg/#pos, capped at 10, the same for every label type); the ablation uses none. **Why it should win:** it learns features end to end at twice the resolution, so it can capture small, local findings that global descriptors average away, and pretraining plus a shared backbone helps rare findings; DenseNet-121 is a standard chest X-ray model [11]. **Progress:** all {{len(RUNS)}} runs are done (3 label types × with/without the weight × 3 seeds; {{f(np.mean([r['train_gpu_minutes'] for r in RUNS.values()]), 1)}} GPU-min each), plus YOLOv8s [12] and an online CPU demo with Grad-CAM.

**Preliminary results.** *H4:* the CNN beats LightGBM (macro-AUROC {{pm(agg('pw','majority','macro_auroc'))}} vs {{f(S.loc['baseline_lgbm','macro_auroc'])}}) on every finding, most clearly among abnormal images ({{f(agg('pw','majority','macro_auroc_within_abnormal').mean())}} vs {{f(S.loc['baseline_lgbm','macro_auroc_within_abnormal'])}}). Contrary to H4, the gain is *larger* on the {{H4['n_rare']}} rare findings ({{f"{H4['mean_gain_rare']:+.3f}"}} vs {{f"{H4['mean_gain_common']:+.3f}"}}). The price is GPU training and {{f(CT['cnn_cpu_ms_per_image_batch1'], 0)}} vs {{f(BT['cpu_ms_per_image_logreg_total'], 0)}} ms per image on CPU. *H1 (Table 2):* with the weight, soft labels are *worse* calibrated than majority labels (ECE {{f(agg('pw','soft','macro_ece').mean())}} vs {{f(agg('pw','majority','macro_ece').mean())}}; paired CI above 0 on {{int((D[(D.variant=='pw') & D.a.str.startswith('cnn_soft')].d_macro_ece_lo > 0).sum())}}/3 seeds), probably because the weight amplifies partial positives. Without it, soft labels have lower ECE on every seed ({{f(agg('nopw','soft','macro_ece').mean())}} vs {{f(agg('nopw','majority','macro_ece').mean())}}; CI below 0 on {{int((D[(D.variant=='nopw') & D.a.str.startswith('cnn_soft')].d_macro_ece_hi < 0).sum())}}/3) at equal AUROC: weak support. *H3* is supported (union: highest sensitivity, lowest specificity, worst ECE and Brier). Temperature scaling barely helps (T = {{f(cnn(S,'pw')['T'].min(), 2)}}–{{f(cnn(S,'pw')['T'].max(), 2)}}; Fig. 3), but per-class Platt scaling brings every CNN to ECE {{f(S[S.kind=='cnn']['macro_ece_platt'].min())}}–{{f(S[S.kind=='cnn']['macro_ece_platt'].max())}}, erasing the label-strategy differences. *H2 (Fig. 4):* without the weight, the soft model is closer to the reader fraction than the majority model (|p − k/3| {{f(agg('nopw','soft','dis_mae_vs_soft_splitvote').mean())}} vs {{f(agg('nopw','majority','dis_mae_vs_soft_splitvote').mean())}}) but ranks images no better (Spearman {{f(agg('nopw','soft','dis_spearman_marked').mean())}} vs {{f(agg('nopw','majority','dis_spearman_marked').mean())}}), and union is closer still ({{f(agg('nopw','union','dis_mae_vs_soft_splitvote').mean())}}): partial support. YOLOv8s reaches mAP@0.4 = {{f(YR['map40'])}}.

**Table 2.** CNN label strategies, mean ± SD over 3 seeds (internal test; AUROC, ECE and Brier are macro averages over the 14 findings).

{{T_LABELS_MAIN}}

# 6. Remaining plan and risks

**Plan to 20 November.** (1) Official test set, if PhysioNet access arrives. (2) Accept a label-strategy difference only if it exceeds the seed spread with a paired CI excluding zero; try calibration-aware epoch selection. (3) *Feature analysis:* ablate the HOG, LBP and intensity groups, inspect model weights and importances, and compare Grad-CAM [13] maps with radiologist boxes. (4) *Complexity analysis:* parameters, training cost, latency and memory for every model, and each finding's AUROC gain against its cost. (5) *Failure analysis* by box size, class frequency, agreement and image quality (blur, noise, downsampling), with a coverage–accuracy curve. (6) YOLO at 1024 px; final report and poster.

**Early failure observations.** (a) Normal-vs-abnormal shortcut: removing normal images drops LightGBM from {{f(S.loc['baseline_lgbm','macro_auroc'])}} to {{f(S.loc['baseline_lgbm','macro_auroc_within_abnormal'])}}. (b) Small or vague findings are hardest: the CNN is weakest on Nodule/Mass ({{f(L.loc['Nodule/Mass','cnn_majority'])}}) and Other lesion ({{f(L.loc['Other lesion','cnn_majority'])}}), and YOLO's AP is near zero for Consolidation ({{f(Y.set_index('class').loc['Consolidation','AP@0.4'])}}) and Other lesion ({{f(Y.set_index('class').loc['Other lesion','AP@0.4'])}}). (c) The 0.5 threshold is poorly placed: sensitivity is {{f(agg('nopw','majority','macro_sens@0.5').mean(), 2)}} without the weight, {{f(agg('pw','majority','macro_sens@0.5').mean(), 2)}} with it but at {{f(agg('pw','majority','macro_ece').mean() / agg('nopw','majority','macro_ece').mean(), 1)}}× the ECE. (d) Model selection is noisy: only {{SC.loc['Pneumothorax','val']}} Pneumothorax validation positives, and ECE with the weight varies {{f(agg('pw','majority','macro_ece').min())}}–{{f(agg('pw','majority','macro_ece').max())}} across seeds.

**Risks.** PhysioNet access may come too late (internal 3-reader labels are noisier); the H1 effect is small; rare-class test sets are tiny ({{int(Y.set_index('class').loc['Pneumothorax','n_gt_boxes'])}} Pneumothorax test boxes); 1024 px YOLO is cut first if GPU time runs short.

{{PAGEBREAK}}

# Figures

![Fig. 1. Reader agreement per finding (image level, all training images). Dark: all 3 readers; light green: 2 of 3; grey: 1 of 3.](../results/fig_disagreement.png){width=70%}

![Fig. 2. Per-finding AUROC on the internal test set, logistic regression vs CNN (majority labels), ordered by training positives.](../results/fig_ladder_auroc.png){width=90%}

![Fig. 3. Reliability diagram, CNN seed 42 with pos_weight (14 findings pooled), before and after temperature scaling.](../results/fig_reliability.png){width=80%}

![Fig. 4. Mean predicted probability by reader agreement on the disagreement set (band: range over seeds). Dashed line: the reader fraction itself.](../results/fig_h2_both.png){width=90%}

{{PAGEBREAK}}

# References

1. H. Q. Nguyen et al., "VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations," *Scientific Data*, 9, 429, 2022.
2. J. C. Peterson, R. M. Battleday, T. L. Griffiths, O. Russakovsky, "Human uncertainty makes classification more robust," *ICCV*, 2019.
3. R. Solovyev, W. Wang, T. Gabruseva, "Weighted boxes fusion: Ensembling boxes from different object detection models," *Image and Vision Computing*, 107, 2021.
4. M. P. Naeini, G. Cooper, M. Hauskrecht, "Obtaining well calibrated probabilities using Bayesian binning," *AAAI*, 2015.
5. C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On calibration of modern neural networks," *ICML*, 2017.
6. J. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," *Advances in Large Margin Classifiers*, 1999.
7. N. Dalal, B. Triggs, "Histograms of oriented gradients for human detection," *CVPR*, 2005.
8. T. Ojala, M. Pietikäinen, T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," *IEEE TPAMI*, 24(7), 2002.
9. G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," *NeurIPS*, 2017.
10. G. Huang, Z. Liu, L. van der Maaten, K. Q. Weinberger, "Densely connected convolutional networks," *CVPR*, 2017.
11. P. Rajpurkar et al., "CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning," arXiv:1711.05225, 2017.
12. G. Jocher, A. Chaurasia, J. Qiu, *Ultralytics YOLOv8*, 2023. https://github.com/ultralytics/ultralytics
13. R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," *ICCV*, 2017.

# Appendix: supporting tables

**Table A1.** Image-level reader agreement, all {{ f"{DS['n_images']:,}" }} training images (number of images that 1, 2 or 3 readers marked).

{{T_STATS}}

**Table A2.** Per-finding AUROC on the internal test set. CNN = majority labels with pos_weight, mean of 3 seeds; gain = CNN − best baseline; "abn. only" = abnormal test images only.

{{T_LADDER}}

**Table A3.** Complexity (CPU: single image; GPU: {{CT.get('gpu', 'RTX PRO 6000')}}).

{{T_COMPLEX}}

**Table A4.** YOLOv8s AP at IoU ≥ 0.4 per class (internal test, fused boxes supported by ≥ 2 readers).

{{T_YOLO}}

**Table A5.** Paired bootstrap differences between label strategies (same seed; positive Δ ECE = worse calibrated than majority). First block: with pos_weight; second: without.

{{T_DIFF_PW}}

{{T_DIFF_NOPW}}
