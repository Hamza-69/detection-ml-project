"""Gradio demo: cardiac cine-MRI segmentation, clinical indices and diagnosis (course project, NOT for clinical use).

    python -m acdc.app --models_dir deploy/models            # local, CPU or GPU
    python -m acdc.app --models_dir deploy/models --public   # deployment mode (binds 0.0.0.0:7860, CPU)

Input: the end-diastolic (ED) and end-systolic (ES) short-axis frames as NIfTI files (as in ACDC), patient height and
weight - or one of the bundled ACDC test patients. Output: segmentation overlays for every slice, LV/RV volumes, ejection
fractions and myocardial mass, the diagnosis of the two-stage pipeline (U-Net -> cardiac features -> logistic
regression) and of the end-to-end CNN (with Grad-CAM), and an uncertainty warning when the segmentation is unreliable.
"""
import argparse
import json
import tempfile
from pathlib import Path

import gradio as gr
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch

from acdc.augment import to_canvas
from acdc.common import keep_largest_component, normalize, probs_to_original, resample_image
from acdc.features import patient_features
from acdc.gradcam import gradcam
from acdc.infer import predict_volume
from acdc.predict_seg import entropy, uncertainty_summary
from acdc.train_cls import CROP, SliceNet
from acdc.unet import UNet2D

BANNER = ("**Course project demo (AUB Machine Learning, Fall 2026) — not a medical device, not for clinical use.** "
          "Models trained on the ACDC dataset (Bernard et al., IEEE TMI 2018; CC BY-NC-SA 4.0).")
COLORS = np.array([[0, 0, 0, 0], [0.2, 0.5, 1, 0.5], [0.1, 0.85, 0.3, 0.5], [1, 0.3, 0.2, 0.5]])


class Models:
    def __init__(self, d: Path, device):
        self.d, self.device = d, device
        self.cfg = json.loads((d / "config.json").read_text())
        self._unet, self._cnn, self._clf = None, None, None

    def unets(self):
        if self._unet is None:
            self._unet = []
            for f in range(5):
                ck = torch.load(self.d / f"unet_f{f}.pt", map_location="cpu", weights_only=False)
                m = UNet2D(in_ch=ck["args"]["in_ch"])
                m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
                m.deep_supervision = False
                self._unet.append(m.to(self.device).eval())
        return self._unet

    def cnns(self):
        if self._cnn is None:
            self._cnn = []
            for f in range(5):
                m = SliceNet(pretrained=False)
                m.load_state_dict({k: v.float() for k, v in torch.load(self.d / f"cnn_f{f}.pt", map_location="cpu")["model"].items()})
                self._cnn.append(m.to(self.device).eval())
        return self._cnn

    def clf(self):
        if self._clf is None:
            self._clf = joblib.load(self.d / "feature_clf.joblib")
        return self._clf


def load_nifti(f):
    im = nib.load(f)
    a = np.asarray(im.dataobj, dtype=np.float32)
    if a.ndim == 4:
        raise gr.Error("Please upload single 3D frames (ED and ES), not the 4D cine file.")
    return a.transpose(2, 0, 1), tuple(float(s) for s in im.header.get_zooms()[:3]), im.affine


def overlay(img, seg, title):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img.T, cmap="gray")
    ax.imshow(COLORS[seg.T], interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def make_app(M: Models, examples_dir: Path):
    groups = M.cfg["groups"]
    names = {"NOR": "normal", "MINF": "previous myocardial infarction", "DCM": "dilated cardiomyopathy",
             "HCM": "hypertrophic cardiomyopathy", "RV": "abnormal right ventricle"}
    ex = sorted(p.name for p in examples_dir.iterdir() if p.is_dir()) if examples_dir.exists() else []

    def run(ed_file, es_file, height, weight, example, ensemble):
        if example:
            dd = examples_dir / example
            info = dict(line.split(": ") for line in (dd / "Info.cfg").read_text().splitlines() if ": " in line)
            ed_file, es_file = dd / f"{example}_ED.nii.gz", dd / f"{example}_ES.nii.gz"
            height, weight = float(info["Height"]), float(info["Weight"])
            truth = f"ACDC reference diagnosis for {example}: **{info['Group']}** ({names.get(info['Group'], '')})"
        else:
            if ed_file is None or es_file is None:
                raise gr.Error("Upload both the ED and the ES frame, or pick an example.")
            truth = ""
        if not height or not weight:
            raise gr.Error("Height and weight are needed (volumes are indexed by body-surface area).")
        unets = M.unets() if ensemble else M.unets()[:1]
        segs, imgs, unc, resampled = {}, {}, {}, {}
        for ph, f in (("ED", ed_file), ("ES", es_file)):
            img, sp, aff = load_nifti(str(f))
            x = torch.from_numpy(resample_image(normalize(img), sp[:2]))
            per_model = [predict_volume(m, x, M.cfg["in_ch"], tta=False, device=M.device) for m in unets]
            p = probs_to_original(torch.stack(per_model).mean(0), img.shape[1:])
            seg = keep_largest_component(p.argmax(1).numpy().astype(np.uint8))
            segs[ph], imgs[ph], resampled[ph] = seg, img, x
            # uncertainty score from the first model alone, like the threshold (single model, no TTA, out-of-fold)
            p0 = probs_to_original(per_model[0], img.shape[1:])
            unc[ph] = uncertainty_summary(seg, entropy(p0).numpy())["ent_fg"]
        if segs["ED"].shape != segs["ES"].shape:
            raise gr.Error("ED and ES frames must have the same geometry.")
        feats = patient_features(segs["ED"], segs["ES"], sp, height, weight)
        prob_feat = M.clf().predict_proba(pd.DataFrame([feats])[M.cfg["features"]].values)[0]

        # end-to-end CNN on [ED, ES, ED-ES] slices + Grad-CAM (mid-ventricular slice)
        ed, es = resampled["ED"], resampled["ES"]
        stack = torch.stack([ed, es, ed - es], 1)
        lo = torch.tensor([float(ed.min()), float(es.min()), 0.0]).view(1, 3, 1, 1)
        stack = (to_canvas(stack - lo, 0.0, CROP) + lo).to(M.device)
        cams, probs = [], []
        for m in (M.cnns() if ensemble else M.cnns()[:1]):
            cam, _, pr = gradcam(m, stack)
            cams.append(cam / (cam.max() + 1e-8)); probs.append(pr)
        cam, prob_cnn = np.mean(cams, 0), np.mean(probs, 0)
        if M.cfg.get("cnn_temperature") and not ensemble:  # T was fitted on single-model out-of-fold outputs
            z = np.log(np.clip(prob_cnn, 1e-7, 1)) / M.cfg["cnn_temperature"]
            prob_cnn = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        z_mid = int(np.argmax((to_canvas(torch.from_numpy(segs["ED"] > 0).float(), 0, CROP)).sum((1, 2))))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(stack[z_mid, 0].cpu().numpy().T, cmap="gray"); ax.imshow(cam[z_mid].T, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set_title("Grad-CAM, end-to-end CNN (ED, mid slice)", fontsize=9); ax.axis("off")
        fig.tight_layout(); fig.canvas.draw(); cam_img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)

        gallery = [(overlay(imgs[ph][z], segs[ph][z], f"{ph} slice {z}"), f"{ph} slice {z}")
                   for ph in ("ED", "ES") for z in range(len(segs[ph]))]
        table = pd.DataFrame([
            ["LV end-diastolic volume", f"{feats['LV_EDV']:.0f} mL ({feats['LV_EDVi']:.0f} mL/m²)"],
            ["LV end-systolic volume", f"{feats['LV_ESV']:.0f} mL ({feats['LV_ESVi']:.0f} mL/m²)"],
            ["LV ejection fraction", f"{feats['LV_EF']:.0f} %"],
            ["RV end-diastolic volume", f"{feats['RV_EDV']:.0f} mL ({feats['RV_EDVi']:.0f} mL/m²)"],
            ["RV end-systolic volume", f"{feats['RV_ESV']:.0f} mL ({feats['RV_ESVi']:.0f} mL/m²)"],
            ["RV ejection fraction", f"{feats['RV_EF']:.0f} %"],
            ["Myocardial mass (ED)", f"{feats['MYO_mass']:.0f} g ({feats['MYO_massi']:.0f} g/m²)"],
            ["Max. wall thickness (ED)", f"{feats['thick_max_ED']:.1f} mm"],
        ], columns=["index", "value"])
        thr = M.cfg["uncertainty_threshold_ent_fg"]
        u = max(unc.values())
        warn = (f"⚠️ **High segmentation uncertainty** (score {u:.3f} > {thr:.3f}, the 90th percentile on training "
                f"patients): check the contours before trusting the numbers.") if u > thr else \
            f"Segmentation uncertainty {u:.3f} (90th percentile on training patients: {thr:.3f})."
        tmp = Path(tempfile.mkdtemp())
        out_files = []
        for ph in ("ED", "ES"):
            pth = tmp / f"segmentation_{ph}.nii.gz"
            nib.save(nib.Nifti1Image(segs[ph].transpose(1, 2, 0).astype(np.uint8), aff), pth)
            out_files.append(str(pth))
        lab = lambda p: {f"{g} ({names[g]})": float(v) for g, v in zip(groups, p)}  # noqa: E731
        return gallery, table, lab(prob_feat), lab(prob_cnn), cam_img, warn + ("\n\n" + truth if truth else ""), out_files

    with gr.Blocks(title="ACDC cardiac MRI demo") as demo:
        gr.Markdown("# Cardiac MRI: segmentation, function and diagnosis\n" + BANNER)
        with gr.Row():
            with gr.Column(scale=1):
                example = gr.Dropdown([""] + ex, value=ex[0] if ex else "", label="ACDC test patient (or leave empty and upload)")
                ed = gr.File(label="ED frame (.nii / .nii.gz)", file_types=[".nii", ".gz"])
                es = gr.File(label="ES frame (.nii / .nii.gz)", file_types=[".nii", ".gz"])
                with gr.Row():
                    h = gr.Number(label="height (cm)", value=170)
                    w = gr.Number(label="weight (kg)", value=70)
                ens = gr.Checkbox(label="5-model ensemble (slower on CPU, slightly more accurate)", value=False)
                btn = gr.Button("Run", variant="primary")
                msg = gr.Markdown()
                files = gr.File(label="predicted segmentations (NIfTI; 1 RV, 2 myocardium, 3 LV)", file_count="multiple")
            with gr.Column(scale=2):
                with gr.Row():
                    lab1 = gr.Label(label="Two-stage pipeline: U-Net → cardiac features → logistic regression", num_top_classes=5)
                    lab2 = gr.Label(label="End-to-end CNN (no segmentation)", num_top_classes=5)
                with gr.Row():
                    table = gr.Dataframe(label="Clinical indices from the predicted segmentation", interactive=False)
                    cam = gr.Image(label="Where the end-to-end CNN looks")
                gal = gr.Gallery(label="Segmentation (blue RV, green myocardium, red LV)", columns=6, height="auto")
        btn.click(run, [ed, es, h, w, example, ens], [gal, table, lab1, lab2, cam, msg, files])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default="deploy/models")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    d = Path(args.models_dir)
    demo = make_app(Models(d, args.device), d / "examples")
    demo.queue(max_size=8).launch(server_name="0.0.0.0" if args.public else "127.0.0.1", server_port=args.port,
                                  share=args.share)


if __name__ == "__main__":
    main()
