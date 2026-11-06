# VinDr-CXR: reader disagreement, label strategies and calibration

AUB Machine Learning course project (Fall 2026) — Mohammad Alaa Al Din Al Hamwi, Abdulrahman Kaddoura, Hamza Rachidi.

**Question.** VinDr-CXR training images were each read by 3 radiologists who often disagree. Does training a chest X-ray
classifier on **soft labels** (fraction of readers who marked a finding) give better-calibrated probabilities than
**majority-vote** or **union** labels, without losing AUROC? Secondary: how much does a fine-tuned CNN (DenseNet-121) gain
over a tuned classical model (HOG/LBP/intensity features + logistic regression / LightGBM), and at what cost?

Demo (CPU, upload-only): https://vindr-cxr-aub.fly.dev/ — *course project demo, not for clinical use.*

## Setup

```bash
uv sync                                   # Python 3.12, torch 2.11 (CUDA 12.8 wheels), see pyproject.toml / uv.lock
```

Data (not redistributed here — VinDr-CXR / VinBigData licence):

```bash
# 1. official labels (accept the competition rules on kaggle.com first)
uv run kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection -f train.csv -p /workspace/data/kaggle
# 2. 512 px PNG conversion of the competition DICOMs (+ original width/height)
uv run kaggle datasets download awsaf49/vinbigdata-512-image-dataset -p /workspace/data/dl_awsaf
```

## Pipeline (run in this order)

| Script | What it does |
|---|---|
| `verify_images.py` | Converts a sample of official competition DICOMs (downloaded with `kaggle competitions download -f train/<id>.dicom`) ourselves and compares them with the PNGs (we found 14/20 bit-identical, the rest ≤ 1 grey level) and checks the original image sizes. |
| `prepare.py` | Counts, per image and finding, how many of the 3 readers drew a box → `votes.csv`; builds union (≥1), majority (≥2) and soft (/3) label files; makes the fixed seed-42 splits (train 60% / val 10% / internal test 15% / disagreement set 15%, stratified on each image's rarest finding); writes the disagreement statistics + figure; decodes all PNGs once into one uint8 array `images_512.npy`. |
| `train_cnn.py` | Fine-tunes ImageNet DenseNet-121 (15 sigmoid outputs) on one label type. The image array is copied to the GPU once and augmented there (small rotation/scale/shift, brightness/contrast, no flips). BCE loss, optional pos_weight, AdamW + cosine, bf16. Keeps the best epoch by validation macro-AUROC and saves logits for val / test / disagreement set. |
| `run_queue.sh` | Runs several `train_cnn.py` jobs one after another (we ran two queues in parallel on one GPU). |
| `features.py`, `baseline.py` | Classical baseline: HOG + LBP + intensity features at 256 px (1,988 values), then one logistic regression (C tuned by 5-fold CV) and one LightGBM (leaves tuned, trees by CV early stopping) per target, on majority labels. |
| `prepare_yolo.py` | Rescales boxes to 512 px, merges the 3 readers' boxes with weighted boxes fusion, counts how many readers support each fused box, keeps those with ≥ 2 readers, writes a YOLO dataset. |
| `train_yolo.py` | Trains YOLOv8s (640 px, 30 epochs, no flips) and computes our own mAP@0.4 per class on the internal test split. |
| `evaluate.py` | All metrics on the internal test set (AUROC + bootstrap CIs, within-abnormal AUROC, ECE, Brier, temperature scaling, sensitivity/specificity) and on the disagreement set (probability per agreement level, Spearman, ECE vs soft targets); writes `results/tables/*.csv`, figures and `results/summary.md`. `--timing` measures inference speed. |
| `export_models.py`, `app.py` | Collects the final weights + temperatures into `deploy/models/`; Gradio demo (probabilities bar chart, abnormal score, Grad-CAM, YOLO boxes; `--public` = upload only). |
| `deploy/` | Dockerfile (CPU-only PyTorch) and `fly.toml` for the Fly.io demo. |

Typical commands:

```bash
uv run python prepare.py --labels_csv /workspace/data/kaggle/train.csv --img_dir /workspace/data/png512/vinbigdata/train \
    --meta_csv /workspace/data/dl_awsaf/vinbigdata/train.csv
uv run python train_cnn.py --labels soft --seed 42                                  # main setting (pos_weight capped at 10)
uv run python train_cnn.py --labels soft --seed 42 --pos_weight_cap 0 --tag _nopw   # ablation without pos_weight
nice -n 10 uv run python baseline.py --threads 6
uv run python prepare_yolo.py && uv run python train_yolo.py
uv run python evaluate.py --timing
uv run python export_models.py --out deploy/models && uv run python app.py --models_dir deploy/models --share
fly deploy . --config deploy/fly.toml --dockerfile deploy/Dockerfile --remote-only --ha=false
```

## Notes

- The official VinDr-CXR test labels (3,000 images, 5-reader consensus) are on PhysioNet behind credentialed access; all
  current results use an **internal test split** of the 15,000 training images.
- Trained weights (preliminary, Milestone 2): [release v0.2-milestone2](https://github.com/Hamza-69/detection-ml-project/releases/tag/v0.2-milestone2).
  `demo_weights.tar.gz` holds the demo models; unpack it with `tar -xzf demo_weights.tar.gz -C deploy/` so they land in
  `deploy/models/`. `all_checkpoints.tar.gz` holds all 18 DenseNet-121 runs and the YOLOv8s checkpoint.
