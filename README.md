# Cardiac MRI (ACDC): segmentation, error propagation and diagnosis

AUB Machine Learning course project (Fall 2026) — Mohammad Alaa Al Din Al Hamwi, Abdulrahman Kaddoura, Hamza Rachidi.

> **Dataset change.** Milestone 2 used VinDr-CXR. The official VinDr-CXR test labels (PhysioNet, credentialed access) could
> not be obtained, so the final project uses the alternative dataset proposed by the course staff: **ACDC** (Automated
> Cardiac Diagnosis Challenge, MICCAI 2017). The VinDr-CXR code and its README are kept in [`vindr/`](vindr/).

**Data.** 150 patients, cine short-axis MRI, evenly split into 5 groups: normal (NOR), previous myocardial infarction
(MINF), dilated cardiomyopathy (DCM), hypertrophic cardiomyopathy (HCM), abnormal right ventricle (RV). Expert contours of
the LV cavity, LV myocardium and RV cavity at end-diastole (ED) and end-systole (ES). Official split: 100 training
patients (20 per group) and 50 test patients (10 per group); the test labels are public since 2022.
Citation: O. Bernard et al., *Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and
Diagnosis: Is the Problem Solved?*, IEEE TMI 37(11), 2018 (CC BY-NC-SA 4.0).

**Question.** In a small dataset (100 training patients), how should a diagnosis system be built: an interpretable
*two-stage* pipeline (segmentation → clinical cardiac indices → classical classifier) or an *end-to-end* CNN? How much does
segmentation quality matter — how do segmentation errors propagate into the clinical indices and into the diagnosis —
and can the system tell when it is wrong?

**Hypotheses.**
- **H1 (segmentation ladder).** A 2D U-Net (nnU-Net recipe) clearly beats a classical per-pixel classifier (hand-crafted
  multiscale features + LightGBM) on Dice and HD95 for all three structures.
- **H2 (error propagation).** Indices from the U-Net segmentation agree closely with the expert ones (Pearson r ≥ 0.95 for
  volumes and LV EF), and diagnosis from predicted segmentations is within 5 points of accuracy of diagnosis from the
  expert segmentations (oracle).
- **H3 (two-stage vs end-to-end).** With 100 training patients, the two-stage pipeline is at least as accurate as an
  end-to-end CNN and better calibrated (NLL / Brier / ECE).
- **H4 (reliability).** (a) Predictive uncertainty flags poor segmentations; (b) data augmentation makes the U-Net more
  robust to image degradations (noise, blur, low resolution, bias field, motion ghosting, contrast); (c) the end-to-end
  CNN's Grad-CAM concentrates on the heart.
- Ablations: 2D vs 2.5D input, Dice+CE vs CE loss, test-time augmentation, connected-component post-processing.

**Protocol.** Five-fold cross-validation on the 100 training patients (stratified by group, seed 42, frozen in
`results/acdc_meta.csv`); every model is trained on 4 folds and predicts the 5th (out-of-fold), so all 100 training
patients get honest predictions. The 50 test patients are used only once, by the ensemble of the 5 fold models. No
early stopping or checkpoint selection on validation data (last-epoch models). Diagnosis classifiers use nested CV.

## Pipeline

| Script (`python -m acdc.<name>`) | What it does |
|---|---|
| `prepare` | Reads all 150 patients, resamples ED/ES images in-plane to 1.25 mm, z-scores intensities, caches them; ground-truth clinical indices; 5-fold split; data summary + example figure. |
| `train_seg --cfg unet2d --fold k` | 2D U-Net (7 stages, 32–512 channels, InstanceNorm, LeakyReLU, deep supervision, 33.5 M parameters), Dice + cross-entropy, SGD Nesterov lr 0.01 poly, 250 epochs × 250 iterations × 32 patches of 256², bf16, `torch.compile`. Patch sampling and nnU-Net-style augmentation (rotation, scaling, mirroring, noise, blur, brightness, contrast, low resolution, gamma) run on the GPU. Ablations: `--in_ch 3` (2.5D), `--loss ce`, `--aug none`. |
| `seg_baseline` | Classical baseline: per-pixel LightGBM on 27 features (Gaussian, gradient and Hessian features at 6 scales + position), class-balanced pixel sampling with prior correction, same folds and post-processing. |
| `predict_seg --cfg unet2d` | Out-of-fold predictions for training patients, 5-fold ensemble for test patients, mirroring TTA, back to the original grid, largest connected component; Dice, HD, HD95, ASSD; uncertainty (entropy, ensemble disagreement). |
| `features`, `classify` | Cardiac features (volumes, EF, mass, BSA-indexed values, RV/LV ratios, wall thickness) from any segmentation; logistic regression and random forest diagnosis, nested CV, scenarios oracle (GT→GT), GT→predicted and predicted→predicted; classical image-feature baseline (no segmentation). |
| `train_cls --fold k --seed s` | End-to-end diagnosis CNN: ImageNet ResNet-18 on every slice as [ED, ES, ED−ES], mean+max pooling over slices, 5 seeds × 5 folds. |
| `evaluate_seg`, `evaluate_cls` | All tables (bootstrap CIs, Wilcoxon / McNemar tests), Bland–Altman plots, slice-position and per-group analysis, uncertainty vs error, coverage–accuracy curves, confusion matrices, temperature scaling. |
| `robustness` | Six degradation types × 4 severities applied to raw test images, full pipeline re-run: Dice and diagnosis accuracy. |
| `gradcam` | Grad-CAM of the end-to-end CNN vs the expert heart contours (energy in heart, enrichment, pointing game). |
| `complexity` | Parameters, FLOPs, training time, inference time per patient on GPU and CPU. |
| `export_models`, `app` | Packages the final models; Gradio demo (upload ED/ES NIfTI or pick an ACDC test patient → contours, indices, both diagnoses, Grad-CAM, uncertainty warning). `deploy/` = Dockerfile + `fly.toml` (CPU). |

Results (tables, figures, per-patient predictions) are in [`results/`](results/); `results/summary.md` collects them.

## Reproduce

```bash
uv sync                                                    # Python 3.14, torch 2.11 (CUDA 12.8), see pyproject.toml / uv.lock
curl -L -o ACDC.zip https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/637218c173e9f0047faa00fb/download
unzip ACDC.zip -d raw                                      # then set ACDC_ROOT=<...>/raw/ACDC/database, ACDC_WORK=<cache dir>
uv run python -m acdc.prepare
for f in 0 1 2 3 4; do uv run python -m acdc.train_seg --cfg unet2d --fold $f; done
uv run python -m acdc.predict_seg --cfg unet2d
uv run python -m acdc.seg_baseline
acdc/run_cls_all.sh                                        # 25 end-to-end CNN runs
uv run python -m acdc.classify --sources unet2d classical_lgbm
uv run python -m acdc.evaluate_seg --cfgs classical_lgbm unet2d
uv run python -m acdc.evaluate_cls --seg_cfgs unet2d classical_lgbm
uv run python -m acdc.robustness && uv run python -m acdc.gradcam && uv run python -m acdc.complexity
uv run python -m acdc.export_models --out deploy/models && uv run python -m acdc.app --models_dir deploy/models
```

`acdc/run_queue.sh` runs a list of GPU jobs one after another (one job saturates the GPU).
