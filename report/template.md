---
title: "Reader disagreement and calibration in chest X-ray classification (VinDr-CXR) — Progress report"
author: "Mohammad Alaa Al Din Al Hamwi, Abdulrahman Kaddoura, Hamza Rachidi"
date: "6 November 2026"
---

*Milestone 2 — progress report and preliminary results. All results are preliminary: one internal test split, at most 12 training epochs, three seeds per configuration. Code: <https://github.com/Hamza-69/detection-ml-project>. Live demo: <https://vindr-cxr-aub.fly.dev/>*

# 1. Research question and changes since the proposal

**Question.** When the three VinDr-CXR readers disagree, does training on *soft labels* (the fraction of readers who marked a finding) give better-calibrated predictions than *majority-vote* or *union* labels, especially on contested images, without lowering AUROC? **Secondary (model ladder):** how much does a fine-tuned CNN gain over a tuned classical model on the same task, for which findings, and at what cost?

**Hypotheses.** **H1** soft labels give lower ECE than majority labels, with macro-AUROC at most 0.01 lower. **H2** on split-vote images, the soft model's probability tracks the reader fraction more closely than the majority model's. **H3** union labels give the highest sensitivity but the worst specificity and calibration. **H4** the CNN beats the classical baseline on macro-AUROC, with a smaller gap on rare classes. We also test whether post-hoc recalibration closes any calibration gap.

**Changes since the 7 October proposal.** (i) The work so far concentrates on the label-strategy experiment and the model ladder; the image-quality, box-size and coverage–accuracy analyses move to the final report. (ii) As proposed, a *disagreement set* (15% of training images) keeps all three votes and is never used for training or tuning. (iii) Baseline and CNN solve the same image-level task (14 findings + "abnormal") with the same metrics; YOLOv8 is evaluated separately. (iv) The official test labels are released only on PhysioNet under credentialed access, which we have not obtained yet; all test numbers therefore use an **internal test split** of the 15,000 training images, labelled by 3-reader majority vote rather than the official 5-reader consensus. (v) Class weighting became an experimental factor: every label strategy is trained with and without a positive-class weight, because the weight itself changes calibration. (vi) Detection uses 640 px (from 512 px images) instead of 1024 px.

# 2. Data and labels

Labels are the official competition `train.csv` (Kaggle VinBigData = VinDr-CXR training set): {{ f"{DS['n_rows']:,}" }} rows, {{ f"{DS['n_images']:,}" }} images, {{DS['n_readers']}} radiologists, {{DS['readers_per_image'][0]}} readers per image. Images are a public 512 × 512 PNG conversion of the competition DICOMs (VOI LUT, MONOCHROME1 inverted); we verified it by converting {{DS['n_verified']}} official DICOMs ourselves: {{DS['n_bit_identical']}} were bit-identical and the others within {{DS['max_gray_diff']}} grey level, and the original sizes used to rescale boxes {{'matched' if DS['sizes_match'] else 'did not all match'}}. A reader *marks* finding *k* if they drew at least one box of class *k*; with n ∈ {0,…,3} votes, **union** = n ≥ 1, **majority** = n ≥ 2 and **soft** = n/3, used directly as the BCE target.

**Table 1.** Image-level reader agreement, all 15,000 training images (bar chart: Appendix Fig. A1).

{{T_STATS}}

Disagreement is the norm: for {{DS['n_findings_over75_split']}} of the 14 findings more than three quarters of the positive images are split votes. The readers **never** disagree on normal vs. abnormal (every image is 0/3 or 3/3 on "any finding"), so the three label strategies differ only in the 14 finding outputs. Some classes are very small under majority vote (Atelectasis {{int(ST.loc['Atelectasis','majority_pos'])}}, Pneumothorax {{int(ST.loc['Pneumothorax','majority_pos'])}} images).

**Splits** (seed 42, fixed before training, stratified on each image's rarest marked finding): train {{ f"{DS['split']['train']:,}" }}, validation {{ f"{DS['split']['val']:,}" }}, internal test {{ f"{DS['split']['test']:,}" }}, disagreement set {{ f"{DS['split']['disagree']:,}" }}. For detection, the readers' boxes are merged by weighted boxes fusion (IoU 0.4) and fused boxes supported by ≥ 2 readers are kept ({{ f"{DS['n_fused_maj']:,}" }} boxes).

# 3. Experimental setup

**Baseline (level 1).** HOG, LBP and intensity-histogram features on 256 px images (1,988 values); per target, an L2 logistic regression (C chosen by 5-fold CV) and a LightGBM model (leaves by CV, trees by CV early stopping), trained on majority labels. **CNN (level 2).** ImageNet DenseNet-121 with 15 sigmoid outputs at 512 px; light GPU augmentation without flips; BCE on the chosen targets; AdamW (lr 10⁻⁴), cosine schedule, 12 epochs, best epoch by validation macro-AUROC. Main setting: positive-class weight = #neg/#pos of the majority train labels, capped at 10 and identical for all label types; ablation: no weight. Each of the six configurations is trained with seeds 42, 43 and 44 ({{len(RUNS)}} runs). **Detector.** YOLOv8s, 640 px, 30 epochs, majority-fused boxes. **Metrics** (internal test, majority labels): macro-AUROC over the 14 findings with 95% bootstrap CIs, also computed among abnormal images only (*which* finding rather than *whether* abnormal); macro ECE (15 bins) and Brier score; temperature scaling (one T per model) and per-class Platt scaling, both fit on validation; sensitivity/specificity at 0.5. On the disagreement set: mean probability per agreement level, mean |p − n/3| on split votes, ECE against the reader fraction, and Spearman correlation with the vote count. Detection: AP at IoU 0.4.

# 4. Preliminary results

**Table 2.** Model ladder: AUROC on the internal test set. CNN = majority labels with pos_weight, mean of {{len(cnn(S,'pw','majority'))}} seeds; gain = CNN − best baseline; "abn. only" = abnormal test images only. Confidence intervals: Appendix Table A1.

{{T_LADDER}}

**Table 3.** Label strategies (CNN), mean ± SD over 3 seeds. ECE and Brier are macro averages over the 14 findings. Per-seed paired differences with CIs: Appendix Table A2.

{{T_LABELS_ALL}}

![Fig. 1. Reliability diagram, CNN seed 42 with pos_weight (14 findings pooled), before and after temperature scaling. Without pos_weight: Appendix Fig. A3.](../results/fig_reliability.png){width=85%}

**Table 4.** Disagreement set ({{ f"{DS['split']['disagree']:,}" }} held-out images, all three votes kept, 14 findings pooled), mean ± SD over 3 seeds. p̄ @k/3 = mean probability on image–finding pairs that k readers marked. Full table with baselines: Appendix Table A4.

{{T_DIS_COMPACT}}

![Fig. 2. Mean predicted probability by reader agreement on the disagreement set (band = range over seeds). Dashed line: the reader fraction itself.](../results/fig_h2_both.png){width=85%}

**Detection.** YOLOv8s reaches **mAP@0.4 = {{f(YR['map40'])}}** on the internal test split (per-class AP: Appendix Table A3), high for large, consistent findings (Aortic enlargement {{f(Y.set_index('class').loc['Aortic enlargement','AP@0.4'])}}, Cardiomegaly {{f(Y.set_index('class').loc['Cardiomegaly','AP@0.4'])}}) and near zero for small or vague ones (Consolidation {{f(Y.set_index('class').loc['Consolidation','AP@0.4'])}}, Other lesion {{f(Y.set_index('class').loc['Other lesion','AP@0.4'])}}).

**Table 5.** Complexity (CPU: single image; GPU: {{CT.get('gpu', 'RTX PRO 6000')}}).

{{T_COMPLEX}}

# 5. What the results suggest so far

**H4 — first part supported, second part not.** The CNN's macro-AUROC ({{pm(agg('pw','majority','macro_auroc'))}}) is well above logistic regression ({{f(S.loc['baseline_logreg','macro_auroc'])}}) and LightGBM ({{f(S.loc['baseline_lgbm','macro_auroc'])}}), and the gap widens among abnormal images only ({{pm(agg('pw','majority','macro_auroc_within_abnormal'))}} vs {{f(S.loc['baseline_lgbm','macro_auroc_within_abnormal'])}}): much of the baselines' score comes from separating normal from abnormal images, whereas the CNN can tell *which* finding is present. Contrary to H4, the gain is larger on rare findings ({{f(H4['mean_gain_rare'])}} on the {{H4['n_rare']}} findings with < 150 training positives vs {{f(H4['mean_gain_common'])}} on the others; Spearman between gain and training positives {{f(H4['spearman_gain_vs_trainpos'],2)}}). The cost is a GPU for training ({{f(np.mean([r['train_gpu_minutes'] for r in RUNS.values()]),1)}} GPU-min per run vs about a CPU-minute for logistic regression) and slower CPU inference ({{f(CT.get('cnn_cpu_ms_per_image_batch1', np.nan),0)}} vs {{f(BT['cpu_ms_per_image_logreg_total'],0)}} ms per image).

**H1 — not supported with pos_weight; weakly supported without it.** With pos_weight, soft labels were *worse* calibrated than majority labels (ECE {{pm(agg('pw','soft','macro_ece'),4)}} vs {{pm(agg('pw','majority','macro_ece'),4)}}; paired CI above zero for {{int((D[(D.variant=='pw') & D.a.str.startswith('cnn_soft')].d_macro_ece_lo > 0).sum())}} of 3 seeds). A plausible cause: under soft targets every image that *any* reader marked contributes positive mass, which the weight multiplies by up to 10. Without pos_weight the direction reverses: soft labels had lower ECE on every seed ({{pm(agg('nopw','soft','macro_ece'),4)}} vs {{pm(agg('nopw','majority','macro_ece'),4)}}; CI below zero for {{int((D[(D.variant=='nopw') & D.a.str.startswith('cnn_soft')].d_macro_ece_hi < 0).sum())}} of 3 seeds), with macro-AUROC unchanged in both settings (all paired differences within ±{{f(D[D.a.str.startswith('cnn_soft')].d_macro_auroc.abs().max(),4)}}). The effect is consistent in sign but small. The weight itself is a trade-off: it raises macro-AUROC ({{pm(agg('pw','majority','macro_auroc'))}} vs {{pm(agg('nopw','majority','macro_auroc'))}} for majority labels) and sensitivity at 0.5, but multiplies ECE by about {{f(agg('pw','majority','macro_ece').mean() / agg('nopw','majority','macro_ece').mean(), 1)}}, and its ECE varies widely between seeds ({{f(agg('pw','majority','macro_ece').min())}}–{{f(agg('pw','majority','macro_ece').max())}}), depending on the selected epoch.

**H3 — supported.** In both settings union labels gave the highest sensitivity (with pos_weight {{pm(agg('pw','union','macro_sens@0.5'),2)}} vs {{pm(agg('pw','majority','macro_sens@0.5'),2)}} for majority), the lowest specificity and the worst ECE and Brier score; its AUROC cost was small with pos_weight ({{pm(agg('pw','union','macro_auroc'))}} vs {{pm(agg('pw','majority','macro_auroc'))}}) and absent without it ({{pm(agg('nopw','union','macro_auroc'))}} vs {{pm(agg('nopw','majority','macro_auroc'))}}).

**H2 — partly supported.** All CNNs become less confident as fewer readers agree (Fig. 2). With pos_weight, soft and majority models track the reader fraction equally (mean |p − k/3| {{pm(agg('pw','soft','dis_mae_vs_soft_splitvote'))}} vs {{pm(agg('pw','majority','dis_mae_vs_soft_splitvote'))}}). Without it, the soft model is closer to the reader fraction than the majority model ({{pm(agg('nopw','soft','dis_mae_vs_soft_splitvote'))}} vs {{pm(agg('nopw','majority','dis_mae_vs_soft_splitvote'))}}) and has the lowest ECE against it ({{pm(agg('nopw','soft','dis_ece_vs_soft'),4)}}), but it does not rank cases by agreement better (Spearman {{pm(agg('nopw','soft','dis_spearman_marked'))}} vs {{pm(agg('nopw','majority','dis_spearman_marked'))}}), and the union model's probabilities are, on average, even closer to the reader fraction on split votes ({{pm(agg('nopw','union','dis_mae_vs_soft_splitvote'))}}). The majority model assigns low probability even to 2/3 cases, which are positives under its own labels.

**Recalibration.** One temperature per model hardly helps (T {{f(cnn(S,'pw')['T'].min(),2)}}–{{f(cnn(S,'pw')['T'].max(),2)}} with pos_weight; Fig. 1), consistent with the weight mainly *shifting* the logits, which a single scale factor cannot undo. **Per-class Platt scaling** removes the shift: every CNN ends at ECE {{f(S[S.kind=='cnn']['macro_ece_platt'].min(),4)}}–{{f(S[S.kind=='cnn']['macro_ece_platt'].max(),4)}}, the same level as the uncalibrated logistic-regression baseline ({{f(S.loc['baseline_logreg','macro_ece'],4)}}). After a cheap per-class recalibration, the training-label choice hardly matters for calibration — the outcome the proposal flagged as possible.

# 6. Issues

- **Data access.** Downloading the official labels required accepting the competition rules; the ~200 GB of DICOMs did not fit on the machine, so we used the verified 512 px conversion. Official test labels need PhysioNet credentialing (Section 1).
- **Class weighting vs calibration.** The capped pos_weight inflates probabilities for rare findings and interacts with the label strategy (H1); we therefore report both settings.
- **Normal-vs-abnormal shortcut.** Per-finding AUROC over all images is inflated by easy normal/abnormal separation (logistic regression reaches {{f(S.loc['baseline_logreg','abnormal_auroc'])}} abnormal AUROC from texture features); the abnormal-only AUROC is the more informative ladder comparison.
- **Noisy model selection.** Some classes have very few validation positives (Pneumothorax {{SC.loc['Pneumothorax','val']}}, Atelectasis {{SC.loc['Atelectasis','val']}}), and selecting epochs by AUROC ignores calibration.
- **Detection resolution.** YOLO was trained at 640 px from 512 px images; small findings are likely under-resolved.

# 7. Plan to 20 November

1. Obtain PhysioNet access and evaluate the final models once on the 3,000 official test images.
2. Keep 3 seeds per configuration; call a difference real only if it exceeds the seed spread and its paired CI excludes zero; select epochs by a calibration-aware criterion as a check.
3. Detection at 1024 px, with AP and recall split by box size.
4. Grad-CAM vs radiologist boxes; baseline coefficients vs Grad-CAM for interpretability.
5. Failure analysis by box size, class frequency, agreement level and image quality (contrast and sharpness bins; controlled blur, noise, downsampling), with augmentation as mitigation.
6. Coverage–accuracy curve (referring the least-confident cases), per label strategy, before and after Platt scaling.
7. Full complexity table and the per-finding "is the CNN worth it" comparison; poster, final report, notebook, and trained models on GitHub.

# References

1. H. Q. Nguyen et al., "VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations," *Scientific Data*, 9, 429, 2022.
2. G. Huang, Z. Liu, L. van der Maaten, K. Q. Weinberger, "Densely connected convolutional networks," *CVPR*, 2017.
3. C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On calibration of modern neural networks," *ICML*, 2017.
4. M. P. Naeini, G. Cooper, M. Hauskrecht, "Obtaining well calibrated probabilities using Bayesian binning," *AAAI*, 2015.
5. J. C. Peterson, R. M. Battleday, T. L. Griffiths, O. Russakovsky, "Human uncertainty makes classification more robust," *ICCV*, 2019.
6. R. Solovyev, W. Wang, T. Gabruseva, "Weighted boxes fusion: Ensembling boxes from different object detection models," *Image and Vision Computing*, 107, 2021.
7. G. Jocher, A. Chaurasia, J. Qiu, *Ultralytics YOLOv8*, 2023. https://github.com/ultralytics/ultralytics
8. R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," *ICCV*, 2017.
9. N. Dalal, B. Triggs, "Histograms of oriented gradients for human detection," *CVPR*, 2005.
10. T. Ojala, M. Pietikäinen, T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," *IEEE TPAMI*, 24(7), 2002.
11. G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," *NeurIPS*, 2017.
12. J. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," *Advances in Large Margin Classifiers*, 1999.

# Appendix

**Table A1.** Macro-AUROC with 95% bootstrap confidence intervals (internal test).

{{T_CI}}

**Table A2.** Paired bootstrap differences between label strategies (same seed; positive Δ ECE = worse calibrated than majority). First block: with pos_weight; second: without.

{{T_DIFF_PW}}

{{T_DIFF_NOPW}}

**Table A3.** YOLOv8s AP at IoU ≥ 0.4 per class (internal test, fused boxes supported by ≥ 2 readers).

{{T_YOLO}}

**Table A4.** Disagreement set, all models including baselines (|p − k/3| also after temperature scaling).

{{T_DIS}}

![Fig. A1. Reader agreement per finding (image level). Dark: all 3 readers; light green: 2 of 3; grey: 1 of 3.](../results/fig_disagreement.png){width=70%}

![Fig. A2. Per-finding AUROC, logistic regression vs CNN (majority labels), ordered by training positives.](../results/fig_ladder_auroc.png){width=95%}

![Fig. A3. Reliability diagram, CNN seed 42 without pos_weight.](../results/fig_reliability_nopw.png){width=85%}
