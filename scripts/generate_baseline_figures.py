import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data/trashcan/mirror"
OUT_DIR = PROJECT_DIR / "baseline_outputs"
FIG_DIR = PROJECT_DIR / "eda_figures"

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

SUPER_COLOURS = {
    "trash": "#d62728",
    "rov": "#6c757d",
    "animal": "#1f77b4",
    "plant": "#2ca02c",
}

def supercat(name):
    n = name.lower()
    if n.startswith("trash"):
        return "trash"
    if n == "rov":
        return "rov"
    if n.startswith("animal"):
        return "animal"
    if n == "plant":
        return "plant"
    return "other"

def savefig(name):
    path = FIG_DIR / f"{name}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print("Saved figure ->", path)

def make_architecture_figure():
    fig, ax = plt.subplots(figsize=(14.2, 4.4))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    boxes = [
        (0.15, 1.05, 2.25, 2.35, "#e8f4f8", "#1f77b4",
         "Underwater Input\n& W8 Pre-Processing\n\n• 480×270 / 360 RGB\n• W7: Flip (p=0.5)\n• W8: Repeat-Factor\n  Sampling (RFS)\n• W8: Red-cast & contrast\n  underwater jitter"),
        (2.70, 1.05, 2.30, 2.35, "#fef5e7", "#d68910",
         "COCO-Pretrained\nCNN Backbones\n\n• W7: MobileNetV3-L\n  (19.0M) & ResNet-50 v1\n• W8: ResNet-50 v2\n  (freeze stages 1–3,\n  fine-tune top stages)"),
        (5.30, 1.05, 2.15, 2.35, "#f4ecf7", "#8e44ad",
         "Feature Pyramid\nNetwork (FPN P2–P6)\n\n• Top-down + lateral\n• W8 FPN v2: extra\n  conv-BN blocks on\n  pyramid outputs for\n  small/murky objects"),
        (7.75, 1.05, 2.15, 2.35, "#e8f8f5", "#16a085",
         "Stage 1 Proposals\nvs Single-Stage\n\n• Faster R-CNN: RPN\n  objectness + deltas\n• W8 Alt (RetinaNet):\n  dense anchors with\n  Focal Loss (γ=2, α=.25)"),
        (10.20, 1.05, 1.85, 2.35, "#fdedec", "#c0392b",
         "Stage 2 RoIAlign\n& Box Head\n\n• 7×7 bilinear pool\n• W7: 2-FC MLP head\n• W8 v2: 4-Conv2d +\n  BN + FC head for\n  sharper localisation"),
        (12.35, 1.05, 1.70, 2.35, "#eafaf1", "#27ae60",
         "Final Detections\n(16 Classes + BG)\n\n• Class probabilities\n• Class-specific box\n  refinement\n• NMS (IoU 0.50)"),
    ]

    for x, y, w, h, fc, ec, text in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor=fc, edgecolor=ec, linewidth=2.0
        )
        ax.add_patch(patch)
        lines = text.split("\n")
        title = "\n".join(lines[:2])
        body = "\n".join(lines[3:])
        ax.text(x + w / 2, y + h - 0.36, title, ha="center", va="center",
                fontsize=10.0, fontweight="bold", color="#1a1a1a")
        ax.text(x + 0.10, y + h - 0.80, body, ha="left", va="top",
                fontsize=8.3, color="#2c3e50", linespacing=1.26)

    arrow_xs = [(2.42, 2.68), (5.02, 5.28), (7.47, 7.73), (9.92, 10.18), (12.07, 12.33)]
    for x1, x2 in arrow_xs:
        ax.annotate(
            "", xy=(x2, 2.2), xytext=(x1, 2.2),
            arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.35", lw=2.2, color="#34495e")
        )

    ax.annotate(
        "Multi-scale FPN features (P2–P5)",
        xy=(11.1, 3.42), xytext=(6.35, 3.72),
        ha="center", va="bottom", fontsize=9, fontweight="bold", color="#8e44ad",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.16", lw=1.8, color="#8e44ad", ls="--")
    )

    loss_patch = FancyBboxPatch(
        (0.15, 0.12), 13.90, 0.62,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor="#f8f9f9", edgecolor="#7f8c8d", linewidth=1.3
    )
    ax.add_patch(loss_patch)
    ax.text(
        7.10, 0.43,
        "Week 7 Baselines: Faster R-CNN (MobileNetV3-FPN & ResNet-50-FPN v1)   |   Week 8 Improvements: (1) Repeat-Factor Sampling + Underwater Colour/Contrast Aug + ResNet-50-FPN v2   (2) RetinaNet-FPN v2 (Focal Loss)",
        ha="center", va="center", fontsize=8.8, fontweight="bold", color="#2c3e50"
    )
    ax.set_title("Week 7 Baseline & Week 8 Improved Detection Pipeline on TrashCan-Material", fontsize=12.8, fontweight="bold", pad=6)
    savefig("09_baseline_architecture_pipeline")

def make_results_figures():
    with open(OUT_DIR / "baseline_results.json", "r") as f:
        data = json.load(f)

    mb = data["models"]["mobilenet_v3"]
    r50 = data["models"]["resnet50"]
    ret = data["models"]["retinanet_r50_v2"]
    r50_v2 = data["models"]["resnet50_v2_improved"]
    lit = data["literature_baselines"]
    err_w7 = data["error_analysis_resnet50_test"]
    err_w8 = data["error_analysis_w8_improved_test"]
    train_counts = data["train_class_counts"]

    # --- Figure 10: Training Loss & Validation AP Progression (Week 7 vs Week 8) ---
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0))
    ax = axes[0]
    for m, col, ls, mk, lbl in [
        (mb, "#5dade2", "--", "s", "W7 Baseline: Faster R-CNN MobileNetV3-FPN"),
        (r50, "#e67e22", "-.", "^", "W7 Baseline: Faster R-CNN ResNet-50-FPN v1"),
        (ret, "#8e44ad", ":", "d", "W8 Alt: RetinaNet ResNet-50-FPN v2 (Focal Loss)"),
        (r50_v2, "#c0392b", "-", "o", "W8 Improved: Faster R-CNN ResNet-50-FPN v2 + RFS + Aug"),
    ]:
        eps = [h["epoch"] for h in m["history"]]
        losses = [h["total_loss"] for h in m["history"]]
        ax.plot(eps, losses, marker=mk, linestyle=ls, lw=2.2, color=col, label=lbl)

    ax.set_title("Training Loss Convergence Across Epochs (Train Split, n=4,989)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Multi-Task Loss per Batch")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=8.2, loc="upper right")

    ax = axes[1]
    for m, col, ls, mk, short_name in [
        (mb, "#5dade2", "--", "s", "W7 MobileNetV3-FPN"),
        (r50, "#e67e22", "-.", "^", "W7 ResNet-50-FPN v1"),
        (ret, "#8e44ad", ":", "d", "W8 RetinaNet-FPN v2"),
        (r50_v2, "#c0392b", "-", "o", "W8 ResNet-50-FPN v2 (RFS+Aug)"),
    ]:
        eps = [h["epoch"] for h in m["history"]]
        v50 = [h["val_AP50"] for h in m["history"]]
        vap = [h["val_AP"] for h in m["history"]]
        ax.plot(eps, v50, marker=mk, linestyle=ls, lw=2.4, color=col,
                label=f"{short_name} Val AP50 (best {m['val']['AP50']:.1f}%, AP {m['val']['AP']:.1f}%)")

    ax.axhline(51.2, color="#333333", ls="--", lw=1.3, label="Hong et al. [1] ResNeXt-101-FPN AP50 (51.2%)")
    ax.set_title("Validation AP50 Progression: Week 7 Baselines vs Week 8 Improvements (Val, n=649)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation COCO AP50 (%)")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=8.0, loc="lower right")

    plt.tight_layout()
    savefig("10_baseline_training_curves")

    # --- Figure 11: COCO Metrics Comparison & Family Breakdown ---
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.2), gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax = axes[0]
    metrics_keys = ["AP", "AP50", "AP75", "APS", "APM", "APL"]
    x = np.arange(len(metrics_keys))
    w = 0.16

    lit_frcnn = lit["Faster R-CNN (ResNeXt-101-FPN) [Hong et al. 2020]"]
    models_plot = [
        ("W7: MobileNetV3-FPN", [mb["test"][k] for k in metrics_keys], "#5dade2"),
        ("W7: ResNet-50-FPN v1", [r50["test"][k] for k in metrics_keys], "#f39c12"),
        ("W8 Alt: RetinaNet-R50-FPN v2", [ret["test"][k] for k in metrics_keys], "#9b59b6"),
        ("W8 Imp: Faster R-CNN R50-FPN v2+RFS+Aug", [r50_v2["test"][k] for k in metrics_keys], "#e74c3c"),
        ("Hong et al. [1]: Faster R-CNN ResNeXt-101", [lit_frcnn[k] for k in metrics_keys], "#34495e"),
    ]

    offsets = [-2 * w, -1 * w, 0.0, 1 * w, 2 * w]
    for (lbl, vals, col), off in zip(models_plot, offsets):
        bars = ax.bar(x + off, vals, w, label=lbl, color=col)
        for bar in bars:
            hv = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, hv + 0.5, f"{hv:.1f}",
                    ha="center", va="bottom", fontsize=6.8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["AP\n@[.50:.95]", "AP50\n(IoU≥0.5)", "AP75\n(IoU≥0.75)", "APS\n(<32² px)", "APM\n(32²–96²)", "APL\n(≥96² px)"])
    ax.set_ylabel("COCO Average Precision (%)")
    ax.set_ylim(0, 66)
    ax.set_title("TrashCan-Material COCO Benchmark: Week 7 Baselines vs Week 8 Models", fontweight="bold")
    ax.legend(fontsize=7.8, loc="upper right")

    ax = axes[1]
    fams = ["rov", "plant", "trash", "animal"]
    fam_labels = ["ROV\n(1 class)", "Plant\n(1 class)", "Trash Debris\n(8 classes)", "Marine Animals\n(6 classes)"]
    fx = np.arange(len(fams))
    fw = 0.25

    for idx_m, (m, lbl, col) in enumerate([
        (mb, "W7 MobileNetV3-FPN", "#5dade2"),
        (r50, "W7 ResNet-50-FPN v1", "#f39c12"),
        (r50_v2, "W8 ResNet-50-FPN v2 (RFS+Aug)", "#e74c3c"),
    ]):
        vals_f = [m["test"]["family"][f]["AP50"] for f in fams]
        bars = ax.bar(fx + (idx_m - 1) * fw, vals_f, fw, label=lbl, color=col)
        for bar in bars:
            hv = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, hv + 0.7, f"{hv:.1f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(fx)
    ax.set_xticklabels(fam_labels)
    ax.set_ylabel("Mean Family Test AP50 (%)")
    ax.set_ylim(0, max(r50_v2["test"]["family"][f]["AP50"] for f in fams) + 12)
    ax.set_title("Family-Level AP50 Gain (Week 7 Baseline -> Week 8 Improved)", fontweight="bold")
    ax.legend(fontsize=8.0, loc="upper right")

    plt.tight_layout()
    savefig("11_baseline_ap_comparison")

    # --- Figure 12: Per-Class AP50 (W7 Baseline vs W8 Improved) & Imbalance Correlation ---
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8), gridspec_kw={"width_ratios": [1.3, 1.0]})
    ax = axes[0]
    pc_w7 = r50["test"]["per_class"]
    pc_w8 = r50_v2["test"]["per_class"]
    sorted_cats = sorted(pc_w8.keys(), key=lambda c: pc_w8[c]["AP50"])
    y_pos = np.arange(len(sorted_cats))
    w7_vals = [pc_w7[c]["AP50"] for c in sorted_cats]
    w8_vals = [pc_w8[c]["AP50"] for c in sorted_cats]
    bar_cols = [SUPER_COLOURS[supercat(c)] for c in sorted_cats]

    ax.barh(y_pos - 0.18, w7_vals, color="#bdc3c7", edgecolor="#7f8c8d", height=0.36, label="W7 Baseline: ResNet-50-FPN v1")
    ax.barh(y_pos + 0.18, w8_vals, color=bar_cols, alpha=0.90, height=0.36, label="W8 Improved: ResNet-50-FPN v2 + RFS + Aug")

    for y_i, (v7, v8) in enumerate(zip(w7_vals, w8_vals)):
        gain = v8 - v7
        sign = "+" if gain >= 0 else ""
        ax.text(max(v7, v8) + 0.8, y_i, f"{v8:.1f}% ({sign}{gain:.1f}%)", va="center", fontsize=7.8, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_cats, fontsize=8.8)
    ax.set_xlabel("Test AP50 (%)")
    ax.set_xlim(0, max(w8_vals + w7_vals) + 22)
    ax.set_title("Per-Class Test AP50: Week 7 Baseline vs Week 8 Improved", fontweight="bold")
    fam_patches = [mpatches.Patch(color="#bdc3c7", label="W7 Baseline (All Families)")] + [
        mpatches.Patch(color=SUPER_COLOURS[k], label=f"W8 {k.capitalize()}") for k in ("trash", "rov", "animal", "plant")
    ]
    ax.legend(handles=fam_patches, loc="lower right", fontsize=8.0)

    ax = axes[1]
    counts_arr = np.array([train_counts[c] for c in sorted_cats], dtype=float)
    w7_arr = np.array(w7_vals, dtype=float)
    w8_arr = np.array(w8_vals, dtype=float)
    log_c = np.log10(counts_arr)
    s7, i7 = np.polyfit(log_c, w7_arr, 1)
    s8, i8 = np.polyfit(log_c, w8_arr, 1)
    r7 = np.corrcoef(log_c, w7_arr)[0, 1]
    r8 = np.corrcoef(log_c, w8_arr)[0, 1]

    for c_name, cnt, a7, a8 in zip(sorted_cats, counts_arr, w7_arr, w8_arr):
        col = SUPER_COLOURS[supercat(c_name)]
        ax.scatter(cnt, a7, color="#95a5a6", marker="^", s=45, alpha=0.75, zorder=3)
        ax.scatter(cnt, a8, color=col, marker="o", s=75, edgecolors="k", linewidth=0.7, zorder=4)
        short_lbl = c_name.replace("trash_", "t_").replace("animal_", "a_")
        ax.annotate(short_lbl, (cnt, a8), xytext=(5, 3), textcoords="offset points", fontsize=7.8)

    x_line = np.logspace(np.log10(counts_arr.min() * 0.85), np.log10(counts_arr.max() * 1.15), 100)
    ax.plot(x_line, s7 * np.log10(x_line) + i7, color="#7f8c8d", ls=":", lw=1.6, label=f"W7 Baseline trend (r = {r7:.2f})")
    ax.plot(x_line, s8 * np.log10(x_line) + i8, color="#c0392b", ls="--", lw=2.0, label=f"W8 Improved trend (r = {r8:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Training Bounding Boxes (log scale)")
    ax.set_ylabel("Test AP50 (%)")
    ax.set_title("Class Imbalance Impact: Week 7 vs Week 8 (RFS + Aug)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    savefig("12_baseline_per_class_ap")

    # --- Figure 13: Qualitative Predictions Grid on Test Split (using Best W8 Improved Model) ---
    preds_npz = np.load(OUT_DIR / "test_predictions_r50_v2.npz", allow_pickle=True)
    test_csv = DATA_DIR / "test" / "metadata.csv"
    by_file = defaultdict(list)
    order = []
    with open(test_csv, "r") as f:
        for r in csv.DictReader(f):
            fn = r["file_name"]
            if fn not in by_file:
                order.append(fn)
            by_file[fn].append((r["label"], float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"])))

    all_cats = sorted(train_counts.keys())
    id2cat = {i + 1: c for i, c in enumerate(all_cats)}
    test_offset = data["splits"]["train"] + data["splits"]["val"]

    candidates = []
    seen_primary = set()
    for idx, fn in enumerate(order):
        iid = test_offset + idx
        p = preds_npz[str(iid)].item()
        keep = p["scores"] >= 0.50
        if keep.sum() == 0 or keep.sum() > 5:
            continue
        gt_objs = by_file[fn]
        gt_lbls = [g[0] for g in gt_objs]
        trash_lbls = [l for l in gt_lbls if l.startswith("trash_")]
        if not trash_lbls:
            continue
        primary = trash_lbls[0]
        pred_lbls = [id2cat[int(c)] for c in p["labels"][keep]]
        if primary in pred_lbls:
            score_top = float(p["scores"][keep].max())
            candidates.append((score_top, primary, iid, fn, gt_objs, p))

    candidates.sort(key=lambda x: -x[0])
    chosen = []
    for cand in candidates:
        if cand[1] not in seen_primary and len(chosen) < 8:
            seen_primary.add(cand[1])
            chosen.append(cand)
    for cand in candidates:
        if len(chosen) >= 8:
            break
        if all(cand[2] != c[2] for c in chosen):
            chosen.append(cand)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, (_, _, iid, fn, gt_objs, p) in zip(axes.flat, chosen[:8]):
        img_path = DATA_DIR / "test" / fn
        with Image.open(img_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        ax.imshow(rgb)
        ax.axis("off")

        for lbl, x1, y1, x2, y2 in gt_objs:
            rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                      edgecolor="#00ff66", linewidth=1.6, linestyle="--")
            ax.add_patch(rect)

        keep = p["scores"] >= 0.50
        for box, sc, cid in zip(p["boxes"][keep], p["scores"][keep], p["labels"][keep]):
            cname = id2cat[int(cid)]
            col = SUPER_COLOURS[supercat(cname)]
            x1, y1, x2, y2 = box
            rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                      edgecolor=col, linewidth=2.2)
            ax.add_patch(rect)
            short_c = cname.replace("trash_", "")
            ax.text(x1, max(12, y1 - 4), f"{short_c} {sc:.2f}",
                    fontsize=7.8, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=col, alpha=0.85, edgecolor="none"))
        ax.set_title(fn, fontsize=8.5)

    fig.suptitle("Faster R-CNN (ResNet-50-FPN v2 + RFS + Underwater Aug) Test Predictions (solid) vs Ground Truth (dashed green)",
                 fontsize=12.2, fontweight="bold", y=0.99)
    plt.tight_layout()
    savefig("13_baseline_predictions_grid")

    # --- Figure 14: Error Analysis & Failure Mode Reduction (Week 7 vs Week 8) ---
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.0), gridspec_kw={"width_ratios": [1.1, 1.05]})
    ax = axes[0]
    fp_pct_w7 = err_w7["fp_pct"]
    fp_pct_w8 = err_w8["fp_pct"]
    fp_labels = list(fp_pct_w8.keys())
    w7_fp_vals = [fp_pct_w7[k] for k in fp_labels]
    w8_fp_vals = [fp_pct_w8[k] for k in fp_labels]
    y_fp = np.arange(len(fp_labels))

    ax.barh(y_fp - 0.18, w7_fp_vals, color="#bdc3c7", height=0.35, label="W7 Baseline (ResNet-50-FPN v1)")
    ax.barh(y_fp + 0.18, w8_fp_vals, color="#e67e22", height=0.35, label="W8 Improved (ResNet-50-FPN v2 + RFS + Aug)")
    for yi, (v7, v8) in enumerate(zip(w7_fp_vals, w8_fp_vals)):
        ax.text(max(v7, v8) + 0.8, yi, f"W8: {v8:.1f}% (W7: {v7:.1f}%)", va="center", fontweight="bold", fontsize=8.0)
    ax.set_yticks(y_fp)
    ax.set_yticklabels(fp_labels, fontsize=8.8)
    ax.set_xlabel("% of False Positive Detections (score ≥ 0.5)")
    ax.set_xlim(0, max(w7_fp_vals + w8_fp_vals) + 22)
    ax.set_ylim(-1.1, len(fp_labels) - 0.35)
    ax.set_title("False Positive Error Distribution (Test Split)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.0)

    ax = axes[1]
    labels_fn = list(err_w8["miss_rate_size"].keys()) + [k.replace("\n", " ") for k in err_w8["miss_rate_cond"].keys()]
    w7_fn = list(err_w7["miss_rate_size"].values()) + list(err_w7["miss_rate_cond"].values())
    w8_fn = list(err_w8["miss_rate_size"].values()) + list(err_w8["miss_rate_cond"].values())
    y_fn = np.arange(len(labels_fn))

    ax.barh(y_fn - 0.18, w7_fn, color="#95a5a6", height=0.35, label="W7 Baseline Miss Rate (%)")
    ax.barh(y_fn + 0.18, w8_fn, color="#27ae60", height=0.35, label="W8 Improved Miss Rate (%)")
    for yi, (v7, v8) in enumerate(zip(w7_fn, w8_fn)):
        diff = v8 - v7
        ax.text(max(v7, v8) + 0.8, yi, f"{v8:.1f}% ({diff:+.1f}%)", va="center", fontweight="bold", fontsize=8.0)
    ax.set_yticks(y_fn)
    ax.set_yticklabels(labels_fn, fontsize=8.5)
    ax.axhline(2.5, color="k", ls="--", lw=0.9, alpha=0.5)
    ax.set_xlabel("False Negative / Miss Rate (% of GT Boxes Missed at IoU≥0.5)")
    ax.set_xlim(0, max(w7_fn + w8_fn) + 22)
    ax.set_ylim(-1.3, len(labels_fn) - 0.35)
    ax.set_title("Miss Rate Reduction by Object Size & Image Quality (W7 -> W8)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.0)

    plt.tight_layout()
    savefig("14_baseline_error_analysis")

if __name__ == "__main__":
    make_architecture_figure()
    if (OUT_DIR / "baseline_results.json").exists():
        make_results_figures()
