---
title: "Reader disagreement and calibration in chest X-ray classification (VinDr-CXR) — Progress report"
author: "Mohammad Alaa Al Din Al Hamwi, Abdulrahman Kaddoura, Hamza Rachidi"
date: "6 November 2026"
---

*Progress report and preliminary results (Milestone 2). All results are preliminary: one internal test split, at most 12 training epochs, three seeds.*

# 1. Research question, hypotheses, and changes since the proposal

**Main question.** When radiologists disagree on the training labels, does training on *soft labels* (the fraction of the three readers who marked a finding) give better-calibrated predictions than training on *majority-vote* or *union* labels — especially on images where the readers disagreed — without lowering discrimination (AUROC)?

**Secondary question (model ladder).** How much does a fine-tuned CNN improve over a tuned classical model on the same image-level task, for which findings, and at what cost?

**Hypotheses.**
**H1** — the soft-label model has lower expected calibration error (ECE) than the majority-vote model, with macro-AUROC at most 0.01 lower.
**H2** — on held-out images where readers split, the soft-label model's probability tracks the fraction of readers who marked the finding more closely than the majority model's.
**H3** — the union-label model has the highest sensitivity but the worst specificity and calibration.
**H4** — the CNN beats the classical baseline on macro-AUROC, with a smaller gap on rare classes.
We also ask whether post-hoc recalibration (temperature scaling, and per-class Platt scaling) closes any calibration gap between label strategies.

**What changed since the 7 October proposal.**

- *Narrowed question.* The work so far concentrates on the label-strategy / calibration experiment and the model ladder. The image-quality, box-size and coverage–accuracy analyses from the proposal are scheduled for the final submission (Section 7).
- *Disagreement set.* As proposed, 15% of the training images (2,250) are held out with all three readers' votes and are never used for training or tuning.
- *One image-level task for both ladder levels.* The classical baseline and the CNN predict the same 15 outputs (14 findings + "abnormal") and are scored with the same metrics. YOLOv8 is evaluated separately for localization.
- *Internal test split instead of the official test set.* The 3,000 official test images' labels are released only on PhysioNet under credentialed access, which we do not have yet [TODO Hamza: state whether the PhysioNet credentialing application has been submitted]. All test numbers here come from an internal test split of 2,250 of the 15,000 training images (Section 3). This means the test labels are majority votes of 3 readers, not the 5-reader consensus of the official test set.
- *Resolution.* Classification uses 512 × 512 images, as proposed. Detection uses the same 512 px images upsampled to 640 px, not 1024 px (compute and time; Section 6).
- *Class weighting became an explicit experimental factor* (Section 6): we train every label strategy with and without a positive-class weight, because the weight directly changes calibration.

# 2. Data and label construction

**Source.** Labels are the official `train.csv` of the Kaggle *VinBigData Chest X-ray Abnormalities Detection* competition (VinDr-CXR training set): {{ f"{DS['n_rows']:,}" }} box/label rows, {{ f"{DS['n_images']:,}" }} images, {{DS['n_readers']}} radiologists, exactly {{DS['readers_per_image'][0]}} readers per image. Images are the public 512 × 512 PNG conversion of the competition DICOMs (Kaggle dataset `awsaf49/vinbigdata-512-image-dataset`; VOI LUT applied, MONOCHROME1 inverted, min–max scaled to 8 bits, resized without preserving aspect ratio). Because this conversion is third-party, we checked it: for {{DS['n_verified']}} randomly chosen images we downloaded the official DICOMs from the competition and converted them ourselves; {{DS['n_bit_identical']}}/{{DS['n_verified']}} PNGs were bit-identical and the remaining {{DS['n_verified'] - DS['n_bit_identical']}} differed by at most {{DS['max_gray_diff']}} grey level, and the original image sizes (used to rescale boxes) {{'matched the DICOM headers in all ' + str(DS['n_verified']) + ' cases' if DS['sizes_match'] else 'did NOT all match'}} (`results/image_verification.csv`, `verify_images.py`).

**Image-level votes.** Reader *r* marks finding *k* on image *i* if *r* drew at least one box of class *k*. The number of votes n(i,k) ∈ {0,1,2,3} gives three label sets: **union** (n ≥ 1), **majority** (n ≥ 2) and **soft** (n/3, used directly as the BCE target). The "abnormal" target uses the same rules on "the reader marked any finding".

**Disagreement statistics** (all 15,000 images; `results/disagreement_stats.csv`, Figure 1).

{{T_STATS}}

"% split vote" is the share of images marked by at least one reader on which the readers did not all agree.

![Reader agreement per finding (image level). Dark: all 3 readers; light green: 2 of 3; grey: 1 of 3.](../results/fig_disagreement.png){width=75%}

Three observations shape the rest of the study. (i) Disagreement is the norm, not the exception: for {{DS['n_findings_over75_split']}} of the 14 findings more than three quarters of the positive images are split votes, and only {{f(100*DS['unanimous_share']['Consolidation'],1)}}% (Consolidation) and {{f(100*DS['unanimous_share']['Atelectasis'],1)}}% (Atelectasis) are unanimous. (ii) The readers **never disagree on normal vs. abnormal**: every image is either 0/3 or 3/3 on "any finding". Union, majority and soft labels are therefore identical for the abnormal output, and the label-strategy comparison can only show up in the 14 per-finding outputs. (iii) Several classes are very small under majority labels (Atelectasis {{int(ST.loc['Atelectasis','majority_pos'])}} and Pneumothorax {{int(ST.loc['Pneumothorax','majority_pos'])}} images in total), so per-class results for them are noisy.

**Box labels (YOLO).** Boxes are rescaled from DICOM pixels to 512 px with the original width/height, merged across readers per finding with weighted boxes fusion (IoU 0.4), and each fused box is assigned the number of distinct readers whose box of that class overlaps it at IoU ≥ 0.4. The majority box set keeps fused boxes supported by ≥ 2 readers: {{ f"{DS['n_fused_maj']:,}" }} boxes out of {{ f"{DS['n_fused']:,}" }} fused boxes (train {{ f"{DS['maj_boxes_by_split']['train']:,}" }} / val {{DS['maj_boxes_by_split']['val']}} / test {{ f"{DS['maj_boxes_by_split']['test']:,}" }}). Agreement on *location* is lower than on *presence*: e.g. Atelectasis is majority-positive on {{int(ST.loc['Atelectasis','majority_pos'])}} images but has only {{DS['maj_boxes_atelectasis']}} majority boxes.

# 3. Experimental setup

**Splits** (`splits.csv`, seed 42, fixed before any training). Each image is stratified on its rarest finding marked by any reader (or "none"): **train {{ f"{DS['split']['train']:,}" }}**, **validation {{ f"{DS['split']['val']:,}" }}**, **internal test {{ f"{DS['split']['test']:,}" }}**, **disagreement set {{ f"{DS['split']['disagree']:,}" }}**. Majority-positive counts per split are in `results/split_counts.csv` (e.g. Pneumothorax {{SC.loc['Pneumothorax','train']}} / {{SC.loc['Pneumothorax','val']}} / {{SC.loc['Pneumothorax','test']}} / {{SC.loc['Pneumothorax','disagree']}}).

**Classical baseline (Level 1).** Features on 256 × 256 images (1,988 values): HOG (9 orientations, 32-px cells, 2 × 2 blocks; 1,764), uniform LBP histograms (P = 8, R = 1) on a 4 × 4 grid (160), and a 32-bin intensity histogram plus per-cell mean/std on the 4 × 4 grid (64). One binary model per target, trained on the train split with majority labels: (a) L2 logistic regression on standardised features, C ∈ {10⁻⁴, 10⁻³, 10⁻², 10⁻¹} chosen per target by 5-fold stratified CV AUROC (selected C = 10⁻³ for {{int((CV.best_param == 'C=0.001').sum())}} targets and 10⁻² for {{int((CV.best_param == 'C=0.01').sum())}}, so the optimum lies inside the grid); (b) LightGBM (learning rate 0.05, num_leaves ∈ {15, 31}, feature fraction 0.3, bagging 0.8, L2 = 1), number of trees chosen by 5-fold CV early stopping.

**CNN (Level 2).** ImageNet-pretrained DenseNet-121 (torchvision), 15 sigmoid outputs, 512 px grayscale input copied to 3 channels with ImageNet normalisation. Augmentation on the GPU: rotation ±7°, scale ±10%, shift ±5%, contrast ±15%, brightness ±0.08; no horizontal flip (left/right anatomy matters). Loss: binary cross-entropy with logits on the chosen targets (soft targets used directly). AdamW (lr 10⁻⁴, weight decay 10⁻⁴), batch 32, 200-step warm-up then cosine decay, 12 epochs, bf16 mixed precision. The checkpoint with the best validation macro-AUROC (against majority labels) is kept; early stopping patience 4. Main setting: positive-class weight = (#neg / #pos) of the *majority* train labels, capped at 10, identical for all three label strategies so that the target is the only difference between runs. Each label strategy is trained with seeds 42, 43 and 44. Ablation: the three strategies retrained without the positive-class weight (plain BCE), {{len(cnn(S,'nopw','majority'))}} seed(s) each.

**Detector.** YOLOv8s (COCO-pretrained, Ultralytics), 640 px input, 30 epochs, batch 32, Ultralytics default augmentation except no horizontal flip, trained on the majority boxes of the train split and validated on val. Evaluation: our own VOC-style all-point AP at IoU ≥ 0.4 per class on the internal test split.

**Metrics** (internal test, ground truth = majority labels). Per-class and macro-AUROC (mean over 14 findings), abnormal-vs-normal AUROC, 95% bootstrap CIs (1,000 resamples of test images); paired bootstrap for differences between label strategies. ECE with 15 equal-width bins per output (macro = mean over findings) and Brier score; temperature scaling with one temperature per model fit on the validation set (BCE vs majority labels), and per-class Platt scaling (logistic regression on each output's logit, fit on the validation set); sensitivity/specificity at 0.5. On the disagreement set: mean predicted probability at each agreement level (0/3 … 3/3), Spearman correlation between probability and reader fraction, ECE against the soft targets, and mean |p − n/3| on split-vote pairs. Because most negatives for any finding are normal images, we also report macro-AUROC **within abnormal images only** (does the model know *which* finding, not just *whether* abnormal).

**Compute.** One NVIDIA RTX PRO 6000 (Blackwell, 96 GB) on RunPod, 13.6 CPU cores (container quota). Two CNN runs shared the GPU at a time, alongside YOLO for part of the time; {{len(RUNS)}} CNN runs in total.

# 4. Preliminary results

All results are on the internal test split ({{ f"{DS['split']['test']:,}" }} images) unless stated otherwise; ground truth is the majority label. "±" is the standard deviation over seeds; brackets are 95% bootstrap CIs. "Abn. only" = AUROC computed only on the abnormal test images.

## 4.1 Model ladder: classical baseline vs CNN (H4)

**Table 1.** Per-finding AUROC. CNN = DenseNet-121 trained on majority labels with pos_weight, mean of {{len(cnn(S,'pw','majority'))}} seeds. Gain = CNN − best baseline. Train pos. = majority-positive training images.

{{T_LADDER}}

**Table 1b.** Macro-AUROC with confidence intervals.

{{T_CI}}

![Per-finding AUROC, logistic-regression baseline vs CNN (majority labels), findings ordered by number of training positives.](../results/fig_ladder_auroc.png){width=95%}

## 4.2 Label strategies: discrimination, calibration, sensitivity (H1, H3, temperature scaling)

**Table 2a.** CNN with pos_weight (main setting), mean ± SD over seeds. ECE and Brier are macro averages over the 14 findings; T = fitted temperature.

{{T_LABELS_PW}}

Paired bootstrap differences on the test set (same seed; positive Δ ECE = worse calibrated than majority):

{{T_DIFF_PW}}

**Table 2b.** Same, CNN trained **without** pos_weight (plain BCE).

{{T_LABELS_NOPW}}

{{T_DIFF_NOPW}}

![Reliability diagram (seed 42, with pos_weight), 14 findings pooled, before and after temperature scaling.](../results/fig_reliability.png){width=95%}

![Reliability diagram (seed 42, without pos_weight).](../results/fig_reliability_nopw.png){width=95%}

## 4.3 Behaviour on the disagreement set (H2)

**Table 3.** Disagreement set ({{ f"{DS['split']['disagree']:,}" }} held-out images, all three votes kept), 14 findings pooled. p̄ @k/3 = mean predicted probability over image–finding pairs that k of 3 readers marked; Spearman over pairs with ≥ 1 vote; |p − k/3| over split-vote pairs (1 or 2 votes); ECE vs soft = ECE against the reader fraction.

{{T_DIS}}

![Mean predicted probability by reader agreement (with pos_weight; band = range over seeds). Dashed: the reader fraction itself.](../results/fig_h2_disagreement.png){width=60%}

![Same, CNN trained without pos_weight.](../results/fig_h2_disagreement_nopw.png){width=60%}

## 4.4 Localization: YOLOv8s (mAP@0.4)

**Table 4.** AP at IoU ≥ 0.4 per class on the internal test split, against fused boxes supported by ≥ 2 readers (train {{YR['epochs']}} epochs at {{YR['imgsz']}} px).

{{T_YOLO}}

## 4.5 Complexity

**Table 5.** Cost of each model (CPU timing: single image; GPU: {{CT.get('gpu', 'RTX PRO 6000')}}). Macro-AUROC from Tables 1–2a.

{{T_COMPLEX}}

# 5. What the results suggest so far

These are preliminary readings: one internal test split, ≤ 12 epochs, {{len(cnn(S,'pw','majority'))}} seeds per configuration, and test labels that are 3-reader majority votes rather than the official 5-reader consensus.

**H4 (CNN vs baseline) — first part supported, second part not.** The CNN's macro-AUROC ({{pm(agg('pw','majority','macro_auroc'))}}) is well above both baselines (logistic regression {{f(S.loc['baseline_logreg','macro_auroc'])}}, LightGBM {{f(S.loc['baseline_lgbm','macro_auroc'])}}), and the gap widens when only abnormal images are scored (CNN {{pm(agg('pw','majority','macro_auroc_within_abnormal'))}} vs {{f(S.loc['baseline_logreg','macro_auroc_within_abnormal'])}} / {{f(S.loc['baseline_lgbm','macro_auroc_within_abnormal'])}}): much of the classical models' score comes from separating normal from abnormal images, while the CNN is much better at telling *which* finding is present. Contrary to H4, the gain is not smaller on rare findings: the mean gain is {{f(H4['mean_gain_rare'])}} on the {{H4['n_rare']}} findings with < 150 training positives vs {{f(H4['mean_gain_common'])}} on the others, and the Spearman correlation between gain and number of training positives is {{f(H4['spearman_gain_vs_trainpos'],2)}}. The price is compute: the baselines train in minutes on a CPU and need no GPU, while each CNN run takes {{f(np.mean([r['train_gpu_minutes'] for r in RUNS.values()]),1)}} GPU-minutes (Table 5).

**H1 (soft labels better calibrated) — not supported with pos_weight; [[H1_NOPW]].** With the positive-class weight, soft labels did not lower ECE: macro-ECE was {{pm(agg('pw','soft','macro_ece'))}} for soft vs {{pm(agg('pw','majority','macro_ece'))}} for majority, and the paired difference was positive (soft worse) for every seed (Table 2a). The AUROC part of H1 holds: soft and majority differ by at most {{f(D[(D.variant=='pw') & D.a.str.startswith('cnn_soft')].d_macro_auroc.abs().max(),4)}} macro-AUROC. The seed-to-seed SD of majority-label ECE ({{f(agg('pw','majority','macro_ece').std(),3)}}) is of the same order as the soft–majority gap, so the size of the difference is not yet reliable even though its sign is consistent. A plausible mechanism: with soft targets every image that *any* reader marked contributes positive mass to the loss, and the pos_weight multiplies that mass by up to 10, which inflates probabilities more than with majority targets.

**H3 (union: most sensitive, least specific, worst calibrated) — supported.** Union labels gave the highest sensitivity at 0.5 ({{pm(agg('pw','union','macro_sens@0.5'))}} vs {{pm(agg('pw','majority','macro_sens@0.5'))}} majority), the lowest specificity ({{pm(agg('pw','union','macro_spec@0.5'))}} vs {{pm(agg('pw','majority','macro_spec@0.5'))}}) and the worst ECE ({{pm(agg('pw','union','macro_ece'))}}) and Brier score, with a small AUROC cost ({{pm(agg('pw','union','macro_auroc'))}} vs {{pm(agg('pw','majority','macro_auroc'))}}).

**H2 (soft model tracks reader fraction on split votes) — [[H2_VERDICT]].** With pos_weight, the mean absolute gap between predicted probability and reader fraction on split-vote pairs was essentially the same for soft ({{pm(agg('pw','soft','dis_mae_vs_soft_splitvote'))}}) and majority ({{pm(agg('pw','majority','dis_mae_vs_soft_splitvote'))}}) labels, and the Spearman correlations were also similar ({{pm(agg('pw','soft','dis_spearman_marked'))}} vs {{pm(agg('pw','majority','dis_spearman_marked'))}}). All CNNs do become less confident as fewer readers agree (Table 3, Figure): the mean probability rises monotonically from 0/3 to 3/3 agreement. [[H2_NOPW]]

**Temperature scaling.** A single temperature per model barely changes calibration here (T between {{f(cnn(S,'pw')['T'].min(),2)}} and {{f(cnn(S,'pw')['T'].max(),2)}} with pos_weight; Table 2a) and does not close the gap between label strategies; for union labels it even raises the macro-ECE over the 14 findings, because one temperature is fit jointly to all 15 outputs. The miscalibration caused by pos_weight is mainly a *shift* in the logits, which a single scale factor cannot remove. **Per-class Platt scaling** (one slope and intercept per output, fit on the validation set) does remove it: macro-ECE falls to {{f(cnn(S,'pw')['macro_ece_platt'].min(),4)}}–{{f(cnn(S,'pw')['macro_ece_platt'].max(),4)}} for every pos_weight model (majority {{pm(agg('pw','majority','macro_ece_platt'),4)}}, soft {{pm(agg('pw','soft','macro_ece_platt'),4)}}, union {{pm(agg('pw','union','macro_ece_platt'),4)}}), the same level as the logistic-regression baseline ({{f(S.loc['baseline_logreg','macro_ece'],4)}} uncalibrated). In other words, after a cheap per-class recalibration the choice of training label hardly matters for calibration, which is the outcome the proposal flagged as possible. [[TS_NOPW]]

**Detection.** YOLOv8s reaches mAP@0.4 = {{f(YR['map40'])}} on the internal test split. It is accurate for large, consistently marked findings (Aortic enlargement {{f(Y.set_index('class').loc['Aortic enlargement','AP@0.4'])}}, Cardiomegaly {{f(Y.set_index('class').loc['Cardiomegaly','AP@0.4'])}}) and poor for small or vaguely defined ones (Consolidation {{f(Y.set_index('class').loc['Consolidation','AP@0.4'])}}, Other lesion {{f(Y.set_index('class').loc['Other lesion','AP@0.4'])}}, Calcification {{f(Y.set_index('class').loc['Calcification','AP@0.4'])}}), several of which have fewer than 20 test boxes. This number is not comparable to the VinBigData leaderboard (different test labels, no "No finding" class).


# 6. Issues and changes

- **Data access.** The official competition labels required accepting the competition rules; the original DICOMs (~200 GB) did not fit on the machine, so we used a public 512 px PNG conversion and verified it against official DICOMs (Section 2). The official 3,000-image test labels need PhysioNet credentialed access, so all test numbers are on an internal split.
- **Class weighting changes calibration.** A positive-class weight of up to 10× deliberately pushes predicted probabilities up for rare findings. This is useful for sensitivity but is in direct tension with calibration, and a single temperature cannot remove a shift of this kind. We therefore report every label strategy with and without the weight (Tables 2a/2b) rather than letting the weight hide or create a soft-vs-majority difference.
- **Normal-vs-abnormal shortcut.** Per-finding AUROC over all test images is inflated by the easy separation of normal from abnormal images (the classical model reaches 0.95 abnormal AUROC from HOG/LBP features alone). The within-abnormal AUROC is the more informative ladder comparison.
- **Calibration depends on the selected epoch.** With pos_weight, ECE varies strongly between seeds of the same configuration (majority labels: {{f(agg('pw','majority','macro_ece').min())}}–{{f(agg('pw','majority','macro_ece').max())}}); the selected epochs were {{', '.join(str(RUNS[n]['best_epoch']) for n in sorted(RUNS) if n.startswith('cnn_majority_s') and not n.endswith('_nopw'))}} for the three majority seeds, and model selection by validation AUROC ignores calibration. Three seeds are therefore the minimum for any calibration claim, and we report seed spread next to every difference.
- **Small validation counts.** Model selection uses validation macro-AUROC, but some classes have very few validation positives (Pneumothorax 3, Atelectasis 6), so selection is noisy for them.
- **Detection resolution.** YOLO was trained at 640 px from 512 px images; small findings (nodules, calcifications) are likely under-resolved.
- [TODO Hamza: anything else that broke or changed that the group wants to mention]

# 7. Plan to 20 November

1. **Official test set:** complete PhysioNet credentialing and evaluate the final models once on the 3,000 official test images (5-reader consensus labels).
2. **Seeds and uncertainty:** keep 3 seeds per configuration; report seed spread next to every soft-vs-majority difference, and call a difference real only if it exceeds the seed spread and its paired bootstrap CI excludes zero.
3. **Detection at 1024 px** (from the original DICOMs or the 1024 px conversion), with recall/AP split by box size.
4. **Grad-CAM vs radiologist boxes:** pointing-game / IoU of Grad-CAM peaks with the fused boxes, CNN vs baseline coefficient maps.
5. **Failure analysis:** errors by box size, by class frequency, by agreement level, and under image-quality degradation (contrast and Laplacian-variance bins on test images; controlled blur, noise and downsampling), with augmentation as mitigation.
6. **Recalibration as a mitigation:** report per-class Platt scaling alongside temperature scaling for every model, and check it on the official test set.
7. **Coverage–accuracy curve:** refer the least-confident cases and report performance on the cases kept, for each label strategy, before/after temperature scaling.
8. **Full complexity analysis** (training GPU-hours, CPU/GPU latency, model size, need for a GPU at deployment) and the per-finding "is the CNN worth it" table.
9. **Poster, final report, notebook and GitHub repository** with trained models available for download.

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

Code: https://github.com/Hamza-69/detection-ml-project
