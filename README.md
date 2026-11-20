# Cardiac MRI (ACDC): segmentation, error propagation and diagnosis

AUB Machine Learning course project (Fall 2026) — Mohammad Alaa Al Din Al Hamwi, Abdulrahman Kaddoura, Hamza Rachidi.

> **Dataset change.** Milestone 2 used VinDr-CXR. The official VinDr-CXR test labels (PhysioNet, credentialed access) could
> not be obtained, so the final project uses the alternative dataset proposed by the course staff: **ACDC** (Automated
> Cardiac Diagnosis Challenge, MICCAI 2017). The VinDr-CXR code and its README are kept on the branch [`vindr-milestone2`](https://github.com/Hamza-69/detection-ml-project/tree/vindr-milestone2/vindr).

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

## Repository layout

| Path | Contents |
|---|---|
| [`ACDC_project.ipynb`](ACDC_project.ipynb) | Course notebook: every step explained, the full source of each module, run cells (off by default), all results |
| [`acdc/`](acdc/) | The pipeline, one module per step (table below) |
| [`results/`](results/) | Every table, figure and per-patient prediction behind the reported numbers; [`summary.md`](results/summary.md) collects them |
| [`deploy/`](deploy/) | CPU Docker image and Fly.io config for the demo (model weights: see [Demo](#demo)) |

## Pipeline

| Script (`python -m acdc.<name>`) | What it does |
|---|---|
| `prepare` | Reads all 150 patients, resamples ED/ES images in-plane to 1.25 mm, z-scores intensities, caches them; ground-truth clinical indices; 5-fold split; data summary + example figure. |
| `train_seg --cfg unet2d --fold k` | 2D U-Net (7 stages, 32–512 channels, InstanceNorm, LeakyReLU, deep supervision, 33.5 M parameters), Dice + cross-entropy, SGD Nesterov lr 0.01 poly, 250 epochs × 250 iterations × 32 patches of 256², bf16, `torch.compile`. Patch sampling and nnU-Net-style augmentation (rotation, scaling, mirroring, noise, blur, brightness, contrast, low resolution, gamma) run on the GPU. Ablations: `--in_ch 3` (2.5D), `--loss ce`, `--aug none`. |
| `tune_seg_baseline` | Hyper-parameter grid for the classical baseline (12 LightGBM configurations, 5-fold CV on training patients only). |
| `seg_baseline` | Classical baseline: per-pixel LightGBM on 27 features (Gaussian, gradient and Hessian features at 6 scales + position), class-balanced pixel sampling with prior correction, same folds and post-processing. |
| `predict_seg --cfg unet2d` | Out-of-fold predictions for training patients, 5-fold ensemble for test patients, mirroring TTA, back to the original grid, largest connected component; Dice, HD, HD95, ASSD; uncertainty (entropy, ensemble disagreement). |
| `features`, `classify` | Cardiac features (volumes, EF, mass, BSA-indexed values, RV/LV ratios, wall thickness) from any segmentation; logistic regression and random forest diagnosis, nested CV, scenarios oracle (GT→GT), GT→predicted and predicted→predicted; classical image-feature baseline (no segmentation). |
| `train_cls --fold k --seed s` | End-to-end diagnosis CNN: ImageNet ResNet-18 on every slice as [ED, ES, ED−ES], mean+max pooling over slices, 5 seeds × 5 folds. |
| `evaluate_seg`, `evaluate_cls` | All tables (bootstrap CIs, Wilcoxon / McNemar tests), Bland–Altman plots, slice-position and per-group analysis, uncertainty vs error, coverage–accuracy curves, confusion matrices, temperature scaling. |
| `robustness` | Six degradation types × 4 severities applied to raw test images, full pipeline re-run: Dice and diagnosis accuracy. |
| `gradcam` | Grad-CAM of the end-to-end CNN vs the expert heart contours (energy in heart, enrichment, pointing game). |
| `split_sensitivity` | Diagnosis results with 10 re-drawn CV fold assignments and 20 random 100/50 train/test splits (how much the conclusions depend on the split). |
| `complexity` | Parameters, FLOPs, training time, inference time per patient on GPU and CPU. |
| `export_models`, `app`, `test_deploy` | Packages the final models; Gradio demo (upload ED/ES NIfTI or pick an ACDC test patient → contours, indices, both diagnoses, Grad-CAM, uncertainty warning). `deploy/` = Dockerfile + `fly.toml` (CPU). `test_deploy` times the live demo. See [Demo](#demo). |


## Demo

**Live:** <https://acdc-cardiac-aub.fly.dev/> (Fly.io, 2 dedicated CPUs / 4 GB, stops when idle). Pick one of five ACDC
test patients (one per group) or upload your own ED and ES frames (`.nii` / `.nii.gz`) with height and weight. It shows
the contours, clinical indices, the diagnosis from both pipelines, a Grad-CAM map and an uncertainty warning.
Measured from a client (`results/deploy_test.json`, made by `python -m acdc.test_deploy`): ~17 s to wake a stopped
machine, ~18 s per patient with one model, ~72 s with the 5-model ensemble. Course project, not for clinical use.

**Trained models:** the release [`v1.0-final`](https://github.com/Hamza-69/detection-ml-project/releases/tag/v1.0-final)
has `acdc_models.tar.gz` (the five fold U-Nets, the five end-to-end CNNs, the diagnosis classifier, `config.json`, and
the five example patients). Run the demo locally without retraining:

```bash
gh release download v1.0-final -R Hamza-69/detection-ml-project -p acdc_models.tar.gz   # or download it from the release page
mkdir -p deploy/models && tar -xzf acdc_models.tar.gz -C deploy/models
uv run python -m acdc.app --models_dir deploy/models               # http://127.0.0.1:7860 (GPU if available)
# or the same container as the live demo (CPU only):
docker build -f deploy/Dockerfile -t acdc-demo . && docker run -p 7860:7860 acdc-demo
```

Redeploy to Fly: `fly deploy . --config deploy/fly.toml --dockerfile deploy/Dockerfile --remote-only --ha=false`.
(On Fly's `shared` CPUs the app was throttled to ~7% of the CPU and one patient took over 10 minutes, so `fly.toml`
uses `performance` CPUs.)

## Reproduce

Hardware used: 1x NVIDIA L40S (46 GB), 8 vCPU, 61 GB RAM. Total compute ~30 GPU-hours (each U-Net fold ~78 min;
5 configurations x 5 folds) plus ~3 CPU-hours. Every command below is exactly what produced `results/`; each step
writes its own files, so steps can be re-run independently. To only re-check the tables and figures from the stored
predictions, skip the training steps (the trained weights are in the release, see [Demo](#demo)).

```bash
# 0) environment and data
uv sync                                    # Python 3.14, torch 2.11 + CUDA 12.8, exact versions in uv.lock
curl -L -o ACDC.zip https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/637218c173e9f0047faa00fb/download
unzip ACDC.zip -d raw
export ACDC_ROOT=$PWD/raw/ACDC/database ACDC_WORK=$PWD/work   # data and cache/checkpoint folders
uv run python -m acdc.prepare              # resampling, cache, frozen 5-fold split (results/acdc_meta.csv)

# 1) U-Net: main model + ablations (5 folds each, GPU)
for f in 0 1 2 3 4; do
  uv run python -m acdc.train_seg --cfg unet2d       --fold $f --epochs 250              # main 2D U-Net
  uv run python -m acdc.train_seg --cfg unet25d      --fold $f --epochs 250 --in_ch 3    # 2.5D input
  uv run python -m acdc.train_seg --cfg unet2d_ce    --fold $f --epochs 250 --loss ce    # CE loss only
  uv run python -m acdc.train_seg --cfg unet2d_noaug --fold $f --epochs 250 --aug none   # no augmentation
done
uv run python -m acdc.predict_seg --cfg unet2d                 # + --no_tta / --no_pp for the two inference ablations
uv run python -m acdc.predict_seg --cfg unet2d --no_tta
uv run python -m acdc.predict_seg --cfg unet2d --no_pp
for c in unet25d unet2d_ce unet2d_noaug; do uv run python -m acdc.predict_seg --cfg $c; done

# 2) classical segmentation baseline (CPU)
uv run python -m acdc.tune_seg_baseline --threads 6            # LightGBM grid (training patients only)
uv run python -m acdc.seg_baseline --threads 6

# 3) diagnosis: end-to-end CNN (25 short GPU runs) and feature-based classifiers
bash acdc/run_cls_all.sh
uv run python -m acdc.classify --sources unet2d classical_lgbm unet25d unet2d_noaug unet2d_ce

# 4) evaluation, failure analysis, complexity, summary
uv run python -m acdc.evaluate_seg --cfgs classical_lgbm unet2d unet25d unet2d_noaug unet2d_ce unet2d_notta unet2d_nopp
uv run python -m acdc.evaluate_cls --seg_cfgs unet2d unet25d classical_lgbm unet2d_noaug unet2d_ce
uv run python -m acdc.robustness --cfgs unet2d unet2d_noaug unet25d
uv run python -m acdc.gradcam
uv run python -m acdc.split_sensitivity
uv run python -m acdc.complexity
uv run python -m acdc.summarize                                # -> results/summary.md

# 5) final models and demo
uv run python -m acdc.export_models --out deploy/models
uv run python -m acdc.app --models_dir deploy/models
```

Notebook: `uv run --with jupyter jupyter lab ACDC_project.ipynb` (set the `RUN` flags in the first code cell to re-run
steps from inside the notebook). Helpers used on our machine, not needed to reproduce: `acdc/run_queue.sh` (runs a
list of GPU jobs one after another) and `acdc/finalize.sh` (re-runs step 4 and commits the results).
