"""Step 6 - Gradio demo (runs on CPU).

  uv run python app.py --models_dir deploy/models              # local, with random internal-test images
  uv run python app.py --models_dir deploy/models --share      # + public gradio.live link
  python app.py --public --device cpu                          # Fly deployment: upload only, no dataset images

Models are loaded lazily the first time they are selected.
"""
import argparse
import json
import os

import cv2
import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FINDINGS = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
]
TARGETS = FINDINGS + ["Abnormal"]
BANNER = ("## VinDr-CXR course project demo — not for clinical use\n"
          "AUB Machine Learning course project (M. A. Al Din Al Hamwi, A. Kaddoura, H. Rachidi). "
          "Image-level probabilities for 14 chest X-ray findings from models trained on VinDr-CXR (512 px).")

ap = argparse.ArgumentParser()
ap.add_argument("--models_dir", default="deploy/models")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--public", action="store_true", help="upload-only mode (no dataset images bundled)")
ap.add_argument("--share", action="store_true", help="create a gradio.live share link")
ap.add_argument("--port", type=int, default=7860)
ap.add_argument("--data_dir", default="/workspace/data")
ARGS = ap.parse_args()
DEV = torch.device(ARGS.device)
TEMPS = json.load(open(f"{ARGS.models_dir}/temperatures.json")) if os.path.exists(f"{ARGS.models_dir}/temperatures.json") else {}
_cache = {}

MODEL_CHOICES = [(f"CNN ({l} labels)", f"cnn_{l}") for l in ["majority", "soft", "union"]
                 if os.path.exists(f"{ARGS.models_dir}/cnn_{l}.pt")]
if os.path.exists(f"{ARGS.models_dir}/logreg.joblib"):
    MODEL_CHOICES.append(("Baseline (logistic regression, HOG+LBP+histogram)", "logreg"))
HAS_YOLO = os.path.exists(f"{ARGS.models_dir}/yolov8s.pt")


# ------------------------------------------------------------------ models (lazy)
def get_cnn(key):
    if key not in _cache:
        import torchvision
        model = torchvision.models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, len(TARGETS))
        sd = torch.load(f"{ARGS.models_dir}/{key}.pt", map_location="cpu")["state_dict"]
        model.load_state_dict({k: v.float() for k, v in sd.items()})
        _cache[key] = model.eval().to(DEV)
    return _cache[key]


def get_logreg():
    if "logreg" not in _cache:
        import joblib
        _cache["logreg"] = joblib.load(f"{ARGS.models_dir}/logreg.joblib")
    return _cache["logreg"]


def get_yolo():
    if "yolo" not in _cache:
        from ultralytics import YOLO
        _cache["yolo"] = YOLO(f"{ARGS.models_dir}/yolov8s.pt")
    return _cache["yolo"]


# ------------------------------------------------------------------ inference
def to_tensor(img512):
    x = torch.from_numpy(img512).float().div(255).view(1, 1, 512, 512).expand(1, 3, 512, 512)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((x - mean) / std).to(DEV)


def predict_probs(key, img512, use_T):
    T = TEMPS.get(key, 1.0) if use_T else 1.0
    if key == "logreg":
        from features import extract_features  # same feature code as training
        f = extract_features(img512)[None]
        p = np.array([m.predict_proba(f)[0, 1] for m in get_logreg()])
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        return 1 / (1 + np.exp(-z / T))
    with torch.no_grad():
        z = get_cnn(key)(to_tensor(img512))[0].float().cpu().numpy()
    return 1 / (1 + np.exp(-z / T))


def gradcam_overlay(key, img512, finding):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    model = get_cnn(key)
    cam = GradCAM(model=model, target_layers=[model.features.denseblock4])
    heat = cam(input_tensor=to_tensor(img512), targets=[ClassifierOutputTarget(TARGETS.index(finding))])[0]
    rgb = cv2.cvtColor(img512, cv2.COLOR_GRAY2RGB)
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return (0.6 * rgb + 0.4 * colored).astype(np.uint8)


def yolo_boxes(img512, conf):
    r = get_yolo().predict(cv2.cvtColor(img512, cv2.COLOR_GRAY2BGR), imgsz=640, conf=conf, verbose=False,
                           device="cpu" if DEV.type == "cpu" else 0)[0]
    out = cv2.cvtColor(img512, cv2.COLOR_GRAY2RGB)
    rows = []
    for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(),
                                      r.boxes.cls.cpu().numpy().astype(int)):
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 60, 60), 2)
        cv2.putText(out, f"{FINDINGS[c]} {s:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        rows.append({"finding": FINDINGS[c], "confidence": round(float(s), 3), "box (x1,y1,x2,y2) @512": f"{x1},{y1},{x2},{y2}"})
    return out, pd.DataFrame(rows, columns=["finding", "confidence", "box (x1,y1,x2,y2) @512"])


def bar_chart(probs, title):
    order = np.argsort(probs[:14])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh([FINDINGS[i] for i in order], probs[order], color="#1f77b4")
    for y, i in enumerate(order):
        ax.text(probs[i] + 0.01, y, f"{probs[i]:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("predicted probability")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def load_upload(img):
    """Any uploaded image -> 512x512 uint8 grayscale (same size as training images)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)


# ------------------------------------------------------------------ dataset browser (non-public mode only)
TEST = None
if not ARGS.public:
    try:
        splits = pd.read_csv(f"{ARGS.data_dir}/splits.csv")
        TEST = {
            "ids": splits[splits.split == "test"].image_id.values,
            "majority": pd.read_csv(f"{ARGS.data_dir}/labels_majority.csv").set_index("image_id"),
            "raw": pd.read_csv(f"{ARGS.data_dir}/kaggle/train.csv"),
            "png_dir": f"{ARGS.data_dir}/png512/vinbigdata/train",
        }
    except FileNotFoundError:
        TEST = None


def random_test_image():
    i = np.random.choice(TEST["ids"])
    img = cv2.imread(f"{TEST['png_dir']}/{i}.png", cv2.IMREAD_GRAYSCALE)
    maj = TEST["majority"].loc[i]
    maj_txt = ", ".join(f for f in FINDINGS if maj[f] == 1) or "No finding"
    raw = TEST["raw"][TEST["raw"].image_id == i]
    readers = []
    for r, g in raw.groupby("rad_id"):
        found = sorted(set(g[g.class_id < 14].class_name))
        readers.append(f"- {r}: {', '.join(found) if found else 'No finding'}")
    info = f"**Image** `{i}` (internal test split)\n\n**Majority label (>= 2 of 3 readers):** {maj_txt}\n\n**Per reader:**\n" + "\n".join(readers)
    return img, info


# ------------------------------------------------------------------ main callback
def run(image, model_key, use_T, finding, yolo_on, yolo_conf):
    if image is None:
        raise gr.Error("Upload an image first" + ("" if ARGS.public else " or pick a random test image") + ".")
    img = load_upload(image)
    probs = predict_probs(model_key, img, use_T)
    title = f"{dict((v, k) for k, v in MODEL_CHOICES)[model_key]}{' (temperature-scaled)' if use_T else ''}"
    abn = f"### Abnormal score: {probs[14]:.3f}\n(probability that at least one finding is present)"
    cam = gradcam_overlay(model_key, img, finding) if model_key.startswith("cnn") else None
    boxes, table = (yolo_boxes(img, yolo_conf) if (yolo_on and HAS_YOLO) else (None, None))
    return bar_chart(probs, title), abn, cam, boxes, table


with gr.Blocks(title="VinDr-CXR demo") as demo:
    gr.Markdown(BANNER)
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(label="Chest X-ray (PNG/JPG)", type="numpy", image_mode="L", height=380)
            if TEST is not None:
                rand_btn = gr.Button("Random internal-test image")
                info = gr.Markdown()
                rand_btn.click(random_test_image, outputs=[image, info])
            model_key = gr.Dropdown(MODEL_CHOICES, value=MODEL_CHOICES[0][1], label="Model")
            use_T = gr.Checkbox(label="Temperature-scaled probabilities (T fit on validation set)", value=False)
            finding = gr.Dropdown(TARGETS, value="Cardiomegaly", label="Grad-CAM finding (CNN models)")
            yolo_on = gr.Checkbox(label="Draw YOLOv8 boxes", value=HAS_YOLO, visible=HAS_YOLO)
            yolo_conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="YOLO confidence threshold", visible=HAS_YOLO)
            go = gr.Button("Run", variant="primary")
        with gr.Column(scale=1):
            plot = gr.Plot(label="Per-finding probabilities")
            abn = gr.Markdown()
        with gr.Column(scale=1):
            cam = gr.Image(label="Grad-CAM (last dense block)", height=380)
            boxes = gr.Image(label="YOLOv8 detections", height=380, visible=HAS_YOLO)
            table = gr.Dataframe(label="Boxes", visible=HAS_YOLO)
    go.click(run, [image, model_key, use_T, finding, yolo_on, yolo_conf], [plot, abn, cam, boxes, table])
    gr.Markdown("Uploaded images are converted to grayscale and resized to 512x512. Probabilities come from models "
                "trained on an internal split of VinDr-CXR and are preliminary. Not a medical device.")

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=ARGS.port, share=ARGS.share)
