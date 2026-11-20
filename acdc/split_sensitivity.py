"""How sensitive are the diagnosis conclusions to the split? (course requirement: "how sensitive the conclusions are to
the train/test split and evaluation design")

Classifier: the main one, multinomial logistic regression with C tuned by inner 5-fold CV (classify.make_model).
Feature sources: gt (expert masks, oracle), unet2d, classical_lgbm (predicted masks), image (no segmentation).

  A) Fold sensitivity: the nested CV on the 100 training patients repeated with 10 new random stratified 5-fold
     assignments (seeds 1-10) instead of the frozen folds.
  B) Train/test split sensitivity: pool all 150 patients and draw 20 random stratified 100/50 splits (seeds 1-20);
     train on 100 (C tuned inside), test on 50. Compared with the official split.

Limitation: the segmentations are NOT re-trained for every split (that would mean re-training 5 U-Nets, ~6.5 GPU-hours
per split). Every patient's predicted masks come from models that never saw that patient (out-of-fold for training
patients, the 5-fold ensemble for test patients), so they are honest, but only the classifier stage is re-split.

Output: results/cls/split_sensitivity.csv (one row per repeat), results/cls/split_sensitivity_summary.csv,
        results/cls/fig_split_sensitivity.png

    python -m acdc.split_sensitivity
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

from acdc.classify import OUT, make_model, metrics, table  # noqa: E402
from acdc.common import GROUPS  # noqa: E402

SOURCES = {"gt": "expert masks (oracle)", "unet2d": "U-Net masks", "classical_lgbm": "classical masks",
           "image": "image features (no segm.)"}
ENC = {g: i for i, g in enumerate(GROUPS)}


def main():
    rows = []
    for src, label in SOURCES.items():
        tab = table(src)
        feats = [c for c in tab.columns if c not in ("pid", "split", "fold", "group")]
        tr = tab[tab.split == "train"].reset_index(drop=True)
        X, y = tr[feats].values, tr.group.map(ENC).values
        # A) nested CV with re-drawn folds
        for seed in range(1, 11):
            oof = np.zeros((len(tr), len(GROUPS)))
            for fit, val in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                oof[val] = make_model("logreg", len(feats)).fit(X[fit], y[fit]).predict_proba(X[val])
            m = metrics(oof, y, n_boot=1)
            rows.append({"protocol": "A: CV, re-drawn folds", "source": src, "label": label, "seed": seed,
                         "acc": m["acc"], "macro_f1": m["macro_f1"], "nll": m["nll"]})
        # B) random 100/50 train/test splits of all 150 patients
        Xall, yall = tab[feats].values, tab.group.map(ENC).values
        for seed in range(1, 21):
            i_tr, i_te = train_test_split(np.arange(len(tab)), test_size=50, stratify=yall, random_state=seed)
            p = make_model("logreg", len(feats)).fit(Xall[i_tr], yall[i_tr]).predict_proba(Xall[i_te])
            m = metrics(p, yall[i_te], n_boot=1)
            rows.append({"protocol": "B: random 100/50 split", "source": src, "label": label, "seed": seed,
                         "acc": m["acc"], "macro_f1": m["macro_f1"], "nll": m["nll"]})
        print(src, "done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "split_sensitivity.csv", index=False)

    # official numbers (frozen folds / official split) from classify.py, for reference
    off = pd.read_csv(OUT / "summary.csv")
    off = off[(off.classifier == "logreg") & (off.train_source == off.test_source)].set_index(["train_source", "eval"])
    s = df.groupby(["protocol", "source", "label"]).agg(acc_mean=("acc", "mean"), acc_sd=("acc", "std"),
                                                         acc_min=("acc", "min"), acc_max=("acc", "max"),
                                                         nll_mean=("nll", "mean"), n=("acc", "size")).reset_index()
    s["acc_official"] = [off.loc[(r.source, "cv" if r.protocol.startswith("A") else "test"), "acc"] for r in s.itertuples()]
    # paired comparison per repeat: U-Net pipeline minus each other source
    piv = df.pivot_table(index=["protocol", "seed"], columns="source", values="acc")
    for other in ("classical_lgbm", "image", "gt"):
        dlt = (piv["unet2d"] - piv[other]).groupby("protocol")
        s.loc[s.source == other, "unet2d_minus_this_mean"] = s.loc[s.source == other, "protocol"].map(dlt.mean())
        s.loc[s.source == other, "unet2d_better_in"] = s.loc[s.source == other, "protocol"].map(
            dlt.apply(lambda v: f"{int((v > 0).sum())}/{len(v)} (ties {int((v == 0).sum())})"))
    s.to_csv(OUT / "split_sensitivity_summary.csv", index=False)
    print(s.round(3).to_string(index=False))

    # figure: accuracy per repeat, official value as a black diamond
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, prot in zip(axes, sorted(df.protocol.unique())):
        d = df[df.protocol == prot]
        data = [d[d.source == k].acc.values for k in SOURCES]
        ax.boxplot(data, widths=0.5)
        for i, k in enumerate(SOURCES, 1):
            ax.scatter(np.full(len(data[i - 1]), i) + np.random.default_rng(i).uniform(-.12, .12, len(data[i - 1])),
                       data[i - 1], s=10, alpha=.6)
            ax.scatter(i, s[(s.protocol == prot) & (s.source == k)].acc_official, marker="D", color="k", zorder=3)
        ax.set_xticks(range(1, 5), [SOURCES[k].replace(" (", "\n(") for k in SOURCES], fontsize=8)
        ax.set_title(prot + (" (10 seeds)" if prot.startswith("A") else " (20 seeds)"), fontsize=10)
        ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("diagnosis accuracy (logistic regression)")
    fig.text(0.5, 0.01, "black diamond = frozen folds / official split", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig_split_sensitivity.png", dpi=150)


if __name__ == "__main__":
    main()
