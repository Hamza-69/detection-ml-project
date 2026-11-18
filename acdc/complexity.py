"""Step 10 - complexity / cost of every model (run when the GPU is idle).

    python -m acdc.complexity
Writes results/complexity.csv: parameters, FLOPs (forward pass), training time, and inference time per patient (ED + ES
volumes, median of repeated runs) on the GPU and on the CPU (4 threads), for one model and for the deployed ensemble.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from acdc.common import RESULTS, WORK
from acdc.infer import load_model, predict_volume
from acdc.seg_baseline import volume_features
from acdc.train_cls import SliceNet, load_stack


def flops(model, x):
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            model(x)
    return fc.get_total_flops()


def timeit(fn, n=5):
    fn()  # warm-up
    ts = []
    for _ in range(n):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    torch.set_num_threads(4)
    pid = "patient101"  # typical test patient (10 slices)
    d = np.load(WORK / "cache" / f"{pid}.npz")
    vols = [torch.from_numpy(d[f"img_{p}"].astype(np.float32)) for p in ("ED", "ES")]
    rows = []
    for cfg in ("unet2d", "unet25d", "unet2d_noaug", "unet2d_ce"):
        if not (WORK / "runs" / f"{cfg}_f4" / "model.pt").exists():
            continue
        mods, cks = zip(*[load_model(WORK / "runs" / f"{cfg}_f{f}" / "model.pt") for f in range(5)])
        in_ch = cks[0]["args"]["in_ch"]
        r = {"model": f"U-Net ({cfg})", "params_M": cks[0]["n_params"] / 1e6,
             "gflops_per_slice_256": flops(mods[0], torch.randn(1, in_ch, 256, 256, device="cuda")) / 1e9,
             "train_gpu_min_per_model": float(np.mean([c["train_minutes"] for c in cks])), "n_models": 5}
        r["infer_gpu_s_1model"] = timeit(lambda: [predict_volume(mods[0], v, in_ch, tta=False) for v in vols])
        r["infer_gpu_s_ensemble_tta"] = timeit(lambda: [predict_volume(list(mods), v, in_ch, tta=True) for v in vols], 3)
        cpu = load_model(WORK / "runs" / f"{cfg}_f0" / "model.pt", "cpu")[0]
        r["infer_cpu_s_1model"] = timeit(lambda: [predict_volume(cpu, v, in_ch, tta=False, device="cpu") for v in vols], 2)
        rows.append(r)
        print(r, flush=True)

    lg = WORK / "runs" / "classical_lgbm.joblib"
    if lg.exists():
        models = joblib.load(lg)
        info = json.loads((RESULTS / "seg" / "classical_lgbm_info.json").read_text())
        m0 = models[0][0]
        feats = [volume_features(v.numpy()) for v in vols]
        r = {"model": "classical pixel classifier (LightGBM)", "params_M": float("nan"),
             "n_trees": int(m0.best_iteration) * 4, "train_gpu_min_per_model": float("nan"),
             "train_cpu_min_all_folds": info["train_minutes"], "n_models": 5}
        r["infer_cpu_s_features"] = timeit(lambda: [volume_features(v.numpy()) for v in vols], 2)
        r["infer_cpu_s_1model"] = r["infer_cpu_s_features"] + timeit(
            lambda: [m0.predict(f.reshape(-1, f.shape[-1]), num_iteration=m0.best_iteration, num_threads=4) for f in feats], 2)
        rows.append(r)
        print(r, flush=True)

    ck0 = WORK / "runs" / "cls_resnet18_s0_f0" / "model.pt"
    if ck0.exists():
        ck = torch.load(ck0, map_location="cpu", weights_only=False)
        m = SliceNet(pretrained=False)
        m.load_state_dict(ck["model"])
        x = load_stack(pid).float()
        mg = m.cuda().eval()
        r = {"model": "end-to-end CNN (ResNet-18 slices)", "params_M": sum(p.numel() for p in m.parameters()) / 1e6,
             "gflops_per_patient": flops(mg, [x.cuda()]) / 1e9, "n_slices": len(x), "n_models": 5,
             "train_gpu_min_per_model": float(np.mean([torch.load(p, map_location="cpu", weights_only=False)["train_minutes"]
                                                       for p in (WORK / "runs").glob("cls_resnet18_s*_f*/model.pt")]))}
        with torch.no_grad():
            r["infer_gpu_s_1model"] = timeit(lambda: mg([x.cuda()]))
            mc = SliceNet(pretrained=False)
            mc.load_state_dict(ck["model"])
            mc.eval()
            r["infer_cpu_s_1model"] = timeit(lambda: mc([x]), 3)
        rows.append(r)
        print(r, flush=True)

    # feature extraction + logistic regression (from a predicted segmentation)
    from acdc.classify import make_model
    from acdc.features import feature_table, patient_features
    ft = feature_table("gt")
    feats = [c for c in ft.columns if c not in ("pid", "split", "fold", "group")]
    tr = ft[ft.split == "train"]
    t = time.perf_counter()
    clf = make_model("logreg", len(feats)).fit(tr[feats].values, tr.group.values)
    fit_s = time.perf_counter() - t
    from acdc.common import load_phase
    _, ged, sp = load_phase(pid, "ED")
    _, ges, _ = load_phase(pid, "ES")
    r = {"model": "cardiac features + logistic regression", "params_M": (len(feats) + 1) * 5 / 1e6,
         "fit_cpu_s_incl_grid_search": fit_s,
         "infer_cpu_s_1model": timeit(lambda: clf.predict_proba(pd.DataFrame([patient_features(ged, ges, sp, 170, 70)])[feats].values))}
    rows.append(r)
    print(r)
    pd.DataFrame(rows).to_csv(RESULTS / "complexity.csv", index=False)


if __name__ == "__main__":
    main()
