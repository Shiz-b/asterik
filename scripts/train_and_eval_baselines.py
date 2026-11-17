import os
import io
import csv
import json
import time
import math
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data/trashcan/mirror"
OUT_DIR = PROJECT_DIR / "baseline_outputs"
FIG_DIR = PROJECT_DIR / "eda_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUPER_COLOURS = {
    "trash": "#d62728",
    "rov": "#7f7f7f",
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

# 1. Load metadata CSVs and image stats
STATS_CSV = PROJECT_DIR / "data/trashcan/image_stats_mirror_material.csv"
size_map = {}
img_quality_map = {}
if STATS_CSV.exists():
    with open(STATS_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = int(row["image_id"])
            size_map[iid] = (int(float(row["width"])), int(float(row["height"])))
            img_quality_map[iid] = {
                "mean_r": float(row["mean_r"]),
                "mean_g": float(row["mean_g"]),
                "mean_b": float(row["mean_b"]),
                "lum_mean": float(row["lum_mean"]),
                "lum_std": float(row["lum_std"]),
            }

def load_splits():
    all_labels = set()
    split_records = {}
    img_id_counter = 0
    for split in ("train", "val", "test"):
        csv_path = DATA_DIR / split / "metadata.csv"
        by_file = defaultdict(list)
        order = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fn = r["file_name"]
                if fn not in by_file:
                    order.append(fn)
                lbl = r["label"]
                all_labels.add(lbl)
                by_file[fn].append((
                    lbl,
                    float(r["x_min"]), float(r["y_min"]),
                    float(r["x_max"]), float(r["y_max"])
                ))
        records = []
        for fn in order:
            w, h = size_map.get(img_id_counter, (480, 270))
            records.append({
                "image_id": img_id_counter,
                "file_name": fn,
                "path": str(DATA_DIR / split / fn),
                "width": w,
                "height": h,
                "objects": by_file[fn],
            })
            img_id_counter += 1
        split_records[split] = records
    classes = sorted(all_labels)
    cat2id = {c: i + 1 for i, c in enumerate(classes)}  # 1..16 (0 is background)
    id2cat = {i + 1: c for i, c in enumerate(classes)}
    return split_records, classes, cat2id, id2cat

print("Loading metadata splits...")
split_records, classes, cat2id, id2cat = load_splits()
NUM_CLASSES = len(classes) + 1  # 16 foreground + 1 background = 17

# 2. Preload JPEG bytes into RAM so GPU training is never bottlenecked by NTFS disk reads
print("Caching JPEG bytes into memory for fast GPU training...")
t_cache = time.time()
all_records = split_records["train"] + split_records["val"] + split_records["test"]
jpeg_cache = {}

def _read_bytes(rec):
    with open(rec["path"], "rb") as f:
        return rec["image_id"], f.read()

with ThreadPoolExecutor(16) as ex:
    for iid, b in ex.map(_read_bytes, all_records):
        jpeg_cache[iid] = b

print(f"Cached {len(jpeg_cache):,} images ({sum(len(b) for b in jpeg_cache.values()) / 1e6:.1f} MB) in {time.time() - t_cache:.1f}s")

class TrashCanDataset(Dataset):
    def __init__(self, records, cat2id, train=False):
        self.records = records
        self.cat2id = cat2id
        self.train = train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        raw_bytes = jpeg_cache[rec["image_id"]]
        with Image.open(io.BytesIO(raw_bytes)) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            arr = np.asarray(rgb, dtype=np.float32) / 255.0

        # Tensor [C, H, W]
        img_tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        boxes = []
        labels = []
        areas = []
        for lbl, x1, y1, x2, y2 in rec["objects"]:
            x1 = max(0.0, min(float(w - 1), x1))
            y1 = max(0.0, min(float(h - 1), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))
            if x2 - x1 >= 1.0 and y2 - y1 >= 1.0:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.cat2id[lbl])
                areas.append((x2 - x1) * (y2 - y1))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            areas_t = torch.tensor(areas, dtype=torch.float32)

        # Random horizontal flip during training
        if self.train and torch.rand(1).item() < 0.5:
            img_tensor = torch.flip(img_tensor, dims=[2])
            if len(boxes_t) > 0:
                old_x1 = boxes_t[:, 0].clone()
                old_x2 = boxes_t[:, 2].clone()
                boxes_t[:, 0] = w - old_x2
                boxes_t[:, 2] = w - old_x1

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([rec["image_id"]], dtype=torch.int64),
            "area": areas_t,
            "iscrowd": torch.zeros((len(boxes_t),), dtype=torch.int64),
        }
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

# 3. Exact MS COCO 101-point Interpolated AP Evaluator
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
REC_THRESHOLDS = np.linspace(0.00, 1.00, 101)
AREA_RANGES = {
    "all": (0.0, 1e10),
    "small": (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large": (96.0 ** 2, 1e10),
}

def compute_coco_metrics(predictions, gt_records, cat2id, id2cat):
    """
    predictions: dict[image_id] -> {'boxes': np.ndarray (N,4), 'scores': np.ndarray (N,), 'labels': np.ndarray (N,)}
    gt_records: list of record dicts
    Returns full COCO metrics dict (AP, AP50, AP75, APS, APM, APL, per_class, family_ap, pr_curves, etc.)
    """
    num_cats = len(cat2id)
    # Organize GT and Detections per (cat_id, image_id)
    gt_by_cat_img = defaultdict(list)
    det_by_cat_img = defaultdict(list)

    for rec in gt_records:
        iid = rec["image_id"]
        w, h = rec["width"], rec["height"]
        for lbl, x1, y1, x2, y2 in rec["objects"]:
            x1 = max(0.0, min(float(w - 1), x1))
            y1 = max(0.0, min(float(h - 1), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))
            if x2 - x1 >= 1.0 and y2 - y1 >= 1.0:
                cid = cat2id[lbl]
                area = (x2 - x1) * (y2 - y1)
                gt_by_cat_img[(cid, iid)].append({"box": [x1, y1, x2, y2], "area": area})

        pred = predictions.get(iid, {"boxes": np.zeros((0, 4)), "scores": np.zeros((0,)), "labels": np.zeros((0,), dtype=int)})
        # Top 100 detections per image
        order = np.argsort(-pred["scores"])[:100]
        boxes = pred["boxes"][order]
        scores = pred["scores"][order]
        labels = pred["labels"][order]
        for b, s, c in zip(boxes, scores, labels):
            area = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
            det_by_cat_img[(int(c), iid)].append({"box": b, "score": float(s), "area": float(area), "image_id": iid})

    img_ids = [r["image_id"] for r in gt_records]

    # Store precision [T=10, R=101, K=num_cats, A=4] and recall [T=10, K=num_cats, A=4]
    area_keys = ["all", "small", "medium", "large"]
    precisions = np.full((len(IOU_THRESHOLDS), len(REC_THRESHOLDS), num_cats, len(area_keys)), -1.0)
    recalls = np.full((len(IOU_THRESHOLDS), num_cats, len(area_keys)), -1.0)

    for k_idx, cid in enumerate(range(1, num_cats + 1)):
        for a_idx, akey in enumerate(area_keys):
            amin, amax = AREA_RANGES[akey]
            n_gt_valid = 0
            all_dets = []
            # Per-image matching
            gt_info_per_img = {}
            for iid in img_ids:
                gts = gt_by_cat_img.get((cid, iid), [])
                if gts:
                    gt_boxes = np.array([g["box"] for g in gts], dtype=np.float32)
                    gt_areas = np.array([g["area"] for g in gts], dtype=np.float32)
                    gt_ignore = (gt_areas < amin) | (gt_areas >= amax)
                    # Sort GTs: non-ignored first, ignored second (standard COCO rule)
                    sort_gt = np.argsort(gt_ignore.astype(int))
                    gt_boxes = gt_boxes[sort_gt]
                    gt_ignore = gt_ignore[sort_gt]
                    n_gt_valid += int((~gt_ignore).sum())
                    gt_info_per_img[iid] = (gt_boxes, gt_ignore)
                else:
                    gt_info_per_img[iid] = (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=bool))

                dets = det_by_cat_img.get((cid, iid), [])
                for d in dets:
                    all_dets.append(d)

            if n_gt_valid == 0:
                continue

            if len(all_dets) == 0:
                precisions[:, :, k_idx, a_idx] = 0.0
                recalls[:, k_idx, a_idx] = 0.0
                continue

            # Sort all detections across dataset by descending score
            all_dets.sort(key=lambda x: -x["score"])
            D = len(all_dets)
            T = len(IOU_THRESHOLDS)
            tp = np.zeros((T, D), dtype=np.float32)
            fp = np.zeros((T, D), dtype=np.float32)

            # Track matched GTs per image per IoU threshold
            gt_matched = {
                iid: np.zeros((T, len(gt_info_per_img[iid][0])), dtype=bool)
                for iid in img_ids
            }

            # Precompute IoU per image if needed, or compute on the fly
            for d_idx, d in enumerate(all_dets):
                iid = d["image_id"]
                d_box = d["box"]
                d_area = d["area"]
                gt_boxes, gt_ignore = gt_info_per_img[iid]

                if len(gt_boxes) == 0:
                    if amin <= d_area < amax:
                        fp[:, d_idx] = 1.0
                    continue

                # Compute IoU between d_box and gt_boxes
                ix1 = np.maximum(gt_boxes[:, 0], d_box[0])
                iy1 = np.maximum(gt_boxes[:, 1], d_box[1])
                ix2 = np.minimum(gt_boxes[:, 2], d_box[2])
                iy2 = np.minimum(gt_boxes[:, 3], d_box[3])
                iw = np.maximum(0.0, ix2 - ix1)
                ih = np.maximum(0.0, iy2 - iy1)
                inter = iw * ih
                gt_a = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                union = gt_a + d_area - inter
                ious = np.where(union > 0, inter / union, 0.0)

                for t_idx, thr in enumerate(IOU_THRESHOLDS):
                    best_iou = thr
                    best_g = -1
                    for g_idx in range(len(gt_boxes)):
                        if gt_matched[iid][t_idx, g_idx] and not gt_ignore[g_idx]:
                            continue
                        # If we already matched a non-ignored GT and now reach ignored GTs, stop if best_g > -1
                        if best_g > -1 and not gt_ignore[best_g] and gt_ignore[g_idx]:
                            break
                        if ious[g_idx] >= best_iou:
                            best_iou = ious[g_idx]
                            best_g = g_idx

                    if best_g == -1:
                        if amin <= d_area < amax:
                            fp[t_idx, d_idx] = 1.0
                    else:
                        if not gt_ignore[best_g]:
                            tp[t_idx, d_idx] = 1.0
                            gt_matched[iid][t_idx, best_g] = True
                        # else matched an ignored GT -> neither TP nor FP

            tp_cum = np.cumsum(tp, axis=1)
            fp_cum = np.cumsum(fp, axis=1)
            rc = tp_cum / float(n_gt_valid)
            pr = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

            for t_idx in range(T):
                r_curve = rc[t_idx]
                p_curve = pr[t_idx].copy()
                recalls[t_idx, k_idx, a_idx] = r_curve[-1] if len(r_curve) > 0 else 0.0
                # Monotonically decreasing envelope
                for i in range(len(p_curve) - 2, -1, -1):
                    if p_curve[i] < p_curve[i + 1]:
                        p_curve[i] = p_curve[i + 1]
                inds = np.searchsorted(r_curve, REC_THRESHOLDS, side="left")
                q = np.zeros((len(REC_THRESHOLDS),), dtype=np.float32)
                valid_inds = inds < len(p_curve)
                q[valid_inds] = p_curve[inds[valid_inds]]
                precisions[t_idx, :, k_idx, a_idx] = q

    def _mean_valid(arr):
        v = arr[arr > -1]
        return float(np.mean(v) * 100.0) if len(v) > 0 else 0.0

    ap_all = _mean_valid(precisions[:, :, :, 0])
    ap50 = _mean_valid(precisions[0, :, :, 0])
    ap75 = _mean_valid(precisions[5, :, :, 0])
    aps = _mean_valid(precisions[:, :, :, 1])
    apm = _mean_valid(precisions[:, :, :, 2])
    apl = _mean_valid(precisions[:, :, :, 3])
    ar100 = _mean_valid(recalls[:, :, 0])

    per_class = {}
    family_ap50_lists = defaultdict(list)
    family_ap_lists = defaultdict(list)
    pr_curves_50 = {}
    for k_idx, cid in enumerate(range(1, num_cats + 1)):
        cname = id2cat[cid]
        c_ap = _mean_valid(precisions[:, :, k_idx, 0])
        c_ap50 = _mean_valid(precisions[0, :, k_idx, 0])
        c_ap75 = _mean_valid(precisions[5, :, k_idx, 0])
        c_ar = _mean_valid(recalls[:, k_idx, 0])
        per_class[cname] = {
            "AP": round(c_ap, 2),
            "AP50": round(c_ap50, 2),
            "AP75": round(c_ap75, 2),
            "AR": round(c_ar, 2),
        }
        fam = supercat(cname)
        family_ap50_lists[fam].append(c_ap50)
        family_ap_lists[fam].append(c_ap)
        pr_curves_50[cname] = [round(float(x), 4) for x in precisions[0, :, k_idx, 0]]

    family_summary = {
        fam: {
            "AP": round(float(np.mean(family_ap_lists[fam])), 2),
            "AP50": round(float(np.mean(family_ap50_lists[fam])), 2),
        }
        for fam in family_ap50_lists
    }

    return {
        "AP": round(ap_all, 2),
        "AP50": round(ap50, 2),
        "AP75": round(ap75, 2),
        "APS": round(aps, 2),
        "APM": round(apm, 2),
        "APL": round(apl, 2),
        "AR100": round(ar100, 2),
        "per_class": per_class,
        "family": family_summary,
        "pr_curves_50": pr_curves_50,
    }

@torch.no_grad()
def evaluate_model(model, loader, gt_records, cat2id, id2cat):
    model.eval()
    preds = {}
    t0 = time.time()
    n_imgs = 0
    for images, targets in loader:
        images_gpu = [img.to(DEVICE, non_blocking=True) for img in images]
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            outputs = model(images_gpu)
        for tgt, out in zip(targets, outputs):
            iid = int(tgt["image_id"].item())
            preds[iid] = {
                "boxes": out["boxes"].detach().cpu().numpy(),
                "scores": out["scores"].detach().cpu().numpy(),
                "labels": out["labels"].detach().cpu().numpy(),
            }
            n_imgs += 1
    elapsed = time.time() - t0
    fps = n_imgs / max(elapsed, 1e-6)
    ms_per_img = 1000.0 / max(fps, 1e-6)
    metrics = compute_coco_metrics(preds, gt_records, cat2id, id2cat)
    metrics["fps"] = round(fps, 1)
    metrics["latency_ms"] = round(ms_per_img, 1)
    return metrics, preds

def build_detector(arch_name, num_classes):
    rpn_kwargs = dict(
        rpn_pre_nms_top_n_train=1000,
        rpn_post_nms_top_n_train=1000,
        rpn_pre_nms_top_n_test=500,
        rpn_post_nms_top_n_test=300,
    )
    if arch_name == "mobilenet_v3":
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
            trainable_backbone_layers=6,
            **rpn_kwargs,
        )
    elif arch_name == "resnet50":
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            trainable_backbone_layers=3,
            **rpn_kwargs,
        )
    else:
        raise ValueError(arch_name)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Match native TrashCan frame scale (480x270 / 480x360)
    model.transform = GeneralizedRCNNTransform(
        min_size=(300,),
        max_size=480,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    return model

# 4. DataLoaders
train_ds = TrashCanDataset(split_records["train"], cat2id, train=True)
val_ds = TrashCanDataset(split_records["val"], cat2id, train=False)
test_ds = TrashCanDataset(split_records["test"], cat2id, train=False)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True, persistent_workers=True)

def train_baseline(arch_name, display_name, epochs=5, max_lr=1.8e-4):
    print(f"\n==========================================")
    print(f"Training {display_name} ({arch_name}) for {epochs} epochs on {DEVICE}...")
    print(f"==========================================")
    model = build_detector(arch_name, NUM_CLASSES).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in params)
    print(f"Total params: {total_params:,} | Trainable params: {trainable_params:,}")

    optimizer = torch.optim.AdamW(params, lr=max_lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=epochs * len(train_loader), pct_start=0.15, div_factor=10.0, final_div_factor=20.0
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    history = []
    best_val_ap = -1.0
    best_ckpt_path = OUT_DIR / f"fasterrcnn_{arch_name}_fpn_best.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        t_ep = time.time()
        loss_sums = defaultdict(float)
        n_batches = 0

        for b_idx, (images, targets) in enumerate(train_loader):
            images_gpu = [img.to(DEVICE, non_blocking=True) for img in images]
            targets_gpu = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                loss_dict = model(images_gpu, targets_gpu)
                losses = sum(loss for loss in loss_dict.values())

            if not math.isfinite(losses.item()):
                lr_scheduler.step()
                continue

            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            loss_sums["total_loss"] += float(losses.item())
            for k, v in loss_dict.items():
                loss_sums[k] += float(v.item())
            n_batches += 1

        train_time = time.time() - t_ep
        avg_losses = {k: round(v / max(1, n_batches), 4) for k, v in loss_sums.items()}

        # Evaluate on Validation split
        val_metrics, _ = evaluate_model(model, val_loader, split_records["val"], cat2id, id2cat)
        ep_record = {
            "epoch": epoch,
            "train_time_s": round(train_time, 1),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            **avg_losses,
            "val_AP": val_metrics["AP"],
            "val_AP50": val_metrics["AP50"],
            "val_AP75": val_metrics["AP75"],
            "val_APS": val_metrics["APS"],
            "val_APM": val_metrics["APM"],
            "val_APL": val_metrics["APL"],
        }
        history.append(ep_record)
        print(
            f"Epoch {epoch}/{epochs} ({train_time:.1f}s) | "
            f"Loss: {avg_losses['total_loss']:.4f} (cls {avg_losses['loss_classifier']:.3f}, box {avg_losses['loss_box_reg']:.3f}, "
            f"rpn_obj {avg_losses['loss_objectness']:.3f}, rpn_box {avg_losses['loss_rpn_box_reg']:.3f}) | "
            f"Val AP: {val_metrics['AP']:.2f} | Val AP50: {val_metrics['AP50']:.2f} | Val AP75: {val_metrics['AP75']:.2f}"
        )

        if val_metrics["AP"] >= best_val_ap:
            best_val_ap = val_metrics["AP"]
            torch.save(model.state_dict(), best_ckpt_path)

    # Load best checkpoint and evaluate on both Val and Test
    model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
    final_val_metrics, val_preds = evaluate_model(model, val_loader, split_records["val"], cat2id, id2cat)
    final_test_metrics, test_preds = evaluate_model(model, test_loader, split_records["test"], cat2id, id2cat)
    print(f"[{display_name} BEST] Val AP={final_val_metrics['AP']:.2f}, AP50={final_val_metrics['AP50']:.2f} | "
          f"Test AP={final_test_metrics['AP']:.2f}, AP50={final_test_metrics['AP50']:.2f}, FPS={final_test_metrics['fps']:.1f}")

    return {
        "arch": arch_name,
        "display_name": display_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "history": history,
        "val": final_val_metrics,
        "test": final_test_metrics,
    }, val_preds, test_preds

# Train both baselines
res_mbv3, val_preds_mbv3, test_preds_mbv3 = train_baseline(
    "mobilenet_v3", "Faster R-CNN (MobileNetV3-Large-FPN)", epochs=5, max_lr=2.5e-4
)
res_r50, val_preds_r50, test_preds_r50 = train_baseline(
    "resnet50", "Faster R-CNN (ResNet-50-FPN)", epochs=5, max_lr=1.5e-4
)

# 5. Error Mode Diagnosis on Test Split (for ResNet-50-FPN)
def analyze_errors(preds, gt_records, cat2id, id2cat, score_thr=0.5):
    """
    Categorize false positives (FP) and false negatives (FN / missed GT) on the test split.
    FP categories:
      - class_confusion_same_family (IoU >= 0.5 with a GT of the same family, e.g. trash_plastic vs trash_etc)
      - class_confusion_cross_family (IoU >= 0.5 with a GT of a different family)
      - localization_error (0.1 <= IoU < 0.5 with same class GT)
      - duplicate_detection (IoU >= 0.5 with an already-matched same-class GT)
      - background_fp (IoU < 0.1 with any GT)
    FN (missed GT) breakdown by size bucket and image quality (murky / colour cast).
    """
    fp_counts = {
        "Within-family class confusion\n(e.g. trash_etc vs trash_plastic)": 0,
        "Cross-family class confusion\n(e.g. trash vs animal/plant)": 0,
        "Localization error\n(0.1 <= IoU < 0.5, right class)": 0,
        "Duplicate box\n(IoU >= 0.5, already matched)": 0,
        "Background false positive\n(rocks, glare, telemetry, IoU < 0.1)": 0,
    }
    fn_by_size = {"small (<32²)": [0, 0], "medium (32²–96²)": [0, 0], "large (≥96²)": [0, 0]}  # [missed, total]
    fn_by_cond = {
        "Normal lighting": [0, 0],
        "Blue/green colour cast\n(red ratio < 0.80)": [0, 0],
        "Low contrast / murky\n(lum std < 30)": [0, 0],
        "Dark frame\n(mean lum < 60)": [0, 0],
    }

    for rec in gt_records:
        iid = rec["image_id"]
        w, h = rec["width"], rec["height"]
        gts = []
        for lbl, x1, y1, x2, y2 in rec["objects"]:
            x1 = max(0.0, min(float(w - 1), x1))
            y1 = max(0.0, min(float(h - 1), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))
            if x2 - x1 >= 1.0 and y2 - y1 >= 1.0:
                gts.append({"box": [x1, y1, x2, y2], "cid": cat2id[lbl], "label": lbl, "area": (x2 - x1) * (y2 - y1)})

        pred = preds.get(iid, {"boxes": np.zeros((0, 4)), "scores": np.zeros((0,)), "labels": np.zeros((0,), dtype=int)})
        keep = pred["scores"] >= score_thr
        p_boxes = pred["boxes"][keep]
        p_scores = pred["scores"][keep]
        p_labels = pred["labels"][keep]
        order = np.argsort(-p_scores)
        p_boxes, p_scores, p_labels = p_boxes[order], p_scores[order], p_labels[order]

        gt_boxes = np.array([g["box"] for g in gts], dtype=np.float32) if gts else np.zeros((0, 4), dtype=np.float32)
        gt_matched = np.zeros((len(gts),), dtype=bool)

        if len(gt_boxes) > 0 and len(p_boxes) > 0:
            ious = box_iou(torch.from_numpy(p_boxes), torch.from_numpy(gt_boxes)).numpy()
        else:
            ious = np.zeros((len(p_boxes), len(gt_boxes)), dtype=np.float32)

        for d_idx in range(len(p_boxes)):
            dcid = int(p_labels[d_idx])
            dlbl = id2cat[dcid]
            if len(gts) == 0:
                fp_counts["Background false positive\n(rocks, glare, telemetry, IoU < 0.1)"] += 1
                continue

            # Check same-class match >= 0.5
            same_cls_mask = np.array([g["cid"] == dcid for g in gts], dtype=bool)
            same_cls_unmatched = same_cls_mask & (~gt_matched)
            if same_cls_unmatched.any():
                masked_ious = np.where(same_cls_unmatched, ious[d_idx], -1.0)
                best_g = int(np.argmax(masked_ious))
                if masked_ious[best_g] >= 0.5:
                    gt_matched[best_g] = True
                    continue  # True Positive!

            # Otherwise it is a False Positive; diagnose why
            max_iou_any = float(np.max(ious[d_idx]))
            best_any_g = int(np.argmax(ious[d_idx]))
            max_iou_same = float(np.max(np.where(same_cls_mask, ious[d_idx], 0.0))) if same_cls_mask.any() else 0.0

            if max_iou_same >= 0.5:
                fp_counts["Duplicate box\n(IoU >= 0.5, already matched)"] += 1
            elif max_iou_same >= 0.1:
                fp_counts["Localization error\n(0.1 <= IoU < 0.5, right class)"] += 1
            elif max_iou_any >= 0.5:
                gt_fam = supercat(gts[best_any_g]["label"])
                pred_fam = supercat(dlbl)
                if gt_fam == pred_fam:
                    fp_counts["Within-family class confusion\n(e.g. trash_etc vs trash_plastic)"] += 1
                else:
                    fp_counts["Cross-family class confusion\n(e.g. trash vs animal/plant)"] += 1
            else:
                fp_counts["Background false positive\n(rocks, glare, telemetry, IoU < 0.1)"] += 1

        # Analyze missed GTs (False Negatives)
        q = img_quality_map.get(iid, {"mean_r": 90, "mean_g": 100, "mean_b": 100, "lum_mean": 90, "lum_std": 40})
        tot_rgb = max(1e-6, q["mean_r"] + q["mean_g"] + q["mean_b"])
        red_ratio = 3.0 * q["mean_r"] / tot_rgb
        is_cast = red_ratio < 0.80
        is_murky = q["lum_std"] < 30.0
        is_dark = q["lum_mean"] < 60.0

        for g_idx, g in enumerate(gts):
            missed = int(not gt_matched[g_idx])
            area = g["area"]
            if area < 32.0 ** 2:
                bkey = "small (<32²)"
            elif area < 96.0 ** 2:
                bkey = "medium (32²–96²)"
            else:
                bkey = "large (≥96²)"
            fn_by_size[bkey][0] += missed
            fn_by_size[bkey][1] += 1

            if is_dark:
                fn_by_cond["Dark frame\n(mean lum < 60)"][0] += missed
                fn_by_cond["Dark frame\n(mean lum < 60)"][1] += 1
            if is_murky:
                fn_by_cond["Low contrast / murky\n(lum std < 30)"][0] += missed
                fn_by_cond["Low contrast / murky\n(lum std < 30)"][1] += 1
            if is_cast:
                fn_by_cond["Blue/green colour cast\n(red ratio < 0.80)"][0] += missed
                fn_by_cond["Blue/green colour cast\n(red ratio < 0.80)"][1] += 1
            if not (is_dark or is_murky or is_cast):
                fn_by_cond["Normal lighting"][0] += missed
                fn_by_cond["Normal lighting"][1] += 1

    total_fp = max(1, sum(fp_counts.values()))
    fp_pct = {k: round(100.0 * v / total_fp, 1) for k, v in fp_counts.items()}
    miss_rate_size = {k: round(100.0 * v[0] / max(1, v[1]), 1) for k, v in fn_by_size.items()}
    miss_rate_cond = {k: round(100.0 * v[0] / max(1, v[1]), 1) for k, v in fn_by_cond.items()}
    return {
        "fp_counts": fp_counts,
        "fp_pct": fp_pct,
        "miss_rate_size": miss_rate_size,
        "miss_rate_cond": miss_rate_cond,
    }

err_analysis = analyze_errors(test_preds_r50, split_records["test"], cat2id, id2cat, score_thr=0.5)
print("Error analysis on Test split:", json.dumps(err_analysis, indent=2))

# Save all numerical results
train_class_counts = defaultdict(int)
for rec in split_records["train"]:
    for lbl, *_ in rec["objects"]:
        train_class_counts[lbl] += 1

full_summary = {
    "dataset": "TrashCan-Material 1.0",
    "splits": {"train": len(split_records["train"]), "val": len(split_records["val"]), "test": len(split_records["test"])},
    "num_classes": len(classes),
    "train_class_counts": dict(train_class_counts),
    "literature_baselines": {
        "Faster R-CNN (ResNeXt-101-FPN) [Hong et al. 2020]": {
            "AP": 29.1, "AP50": 51.2, "AP75": 27.8, "APS": 28.2, "APM": 30.2, "APL": 40.0
        },
        "Mask R-CNN (ResNeXt-101-FPN) [Hong et al. 2020]": {
            "AP": 28.2, "AP50": 54.0, "AP75": 25.6, "APS": 24.1, "APM": 28.7, "APL": 41.8
        },
    },
    "models": {
        "mobilenet_v3": res_mbv3,
        "resnet50": res_r50,
    },
    "error_analysis_resnet50_test": err_analysis,
}

with open(OUT_DIR / "baseline_results.json", "w") as f:
    json.dump(full_summary, f, indent=2)
print("Saved ->", OUT_DIR / "baseline_results.json")

# Save predictions for qualitative plotting
np.savez_compressed(
    OUT_DIR / "test_predictions_r50.npz",
    **{str(k): v for k, v in test_preds_r50.items()}
)
print("Done training and evaluating baselines!")
