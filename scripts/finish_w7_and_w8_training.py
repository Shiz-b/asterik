import os
import io
import gc
import csv
import json
import time
import math
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageEnhance

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.ops import box_iou
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data/trashcan/mirror"
OUT_DIR = PROJECT_DIR / "baseline_outputs"
FIG_DIR = PROJECT_DIR / "eda_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

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

# Load metadata CSVs and image stats
STATS_CSV = PROJECT_DIR / "data/trashcan/image_stats_mirror_material.csv"
size_map = {}
img_quality_map = {}
if STATS_CSV.exists():
    with open(STATS_CSV, "r") as f:
        for row in csv.DictReader(f):
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
            for r in csv.DictReader(f):
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
    cat2id = {c: i + 1 for i, c in enumerate(classes)}
    id2cat = {i + 1: c for i, c in enumerate(classes)}
    return split_records, classes, cat2id, id2cat

print("Loading metadata splits...")
split_records, classes, cat2id, id2cat = load_splits()
NUM_CLASSES = len(classes) + 1  # 17

print("Caching JPEG bytes into memory...")
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
    def __init__(self, records, cat2id, train=False, underwater_aug=False):
        self.records = records
        self.cat2id = cat2id
        self.train = train
        self.underwater_aug = underwater_aug

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        raw_bytes = jpeg_cache[rec["image_id"]]
        with Image.open(io.BytesIO(raw_bytes)) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            if self.train and self.underwater_aug:
                # Contrast and brightness augmentation
                if torch.rand(1).item() < 0.6:
                    cf = float(torch.empty(1).uniform_(0.85, 1.35).item())
                    rgb = ImageEnhance.Contrast(rgb).enhance(cf)
                if torch.rand(1).item() < 0.6:
                    bf = float(torch.empty(1).uniform_(0.88, 1.20).item())
                    rgb = ImageEnhance.Brightness(rgb).enhance(bf)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0

        if self.train and self.underwater_aug and torch.rand(1).item() < 0.5:
            # Underwater RGB colour cast augmentation
            gains = np.array([
                float(torch.empty(1).uniform_(0.95, 1.22).item()),
                float(torch.empty(1).uniform_(0.90, 1.08).item()),
                float(torch.empty(1).uniform_(0.90, 1.08).item()),
            ], dtype=np.float32)
            arr = np.clip(arr * gains[None, None, :], 0.0, 1.0)

        img_tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        boxes, labels, areas = [], [], []
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

def build_repeat_factor_sampler(records, threshold_t=0.04):
    # Repeat-Factor Sampling (RFS) weights for rare classes
    n_imgs = len(records)
    cat_img_counts = defaultdict(int)
    for rec in records:
        cats_in_img = set(lbl for lbl, *_ in rec["objects"])
        for c in cats_in_img:
            cat_img_counts[c] += 1
    cat_freq = {c: cnt / float(n_imgs) for c, cnt in cat_img_counts.items()}
    cat_repeat = {c: max(1.0, math.sqrt(threshold_t / max(f, 1e-6))) for c, f in cat_freq.items()}
    img_weights = []
    for rec in records:
        cats_in_img = set(lbl for lbl, *_ in rec["objects"])
        w = max((cat_repeat[c] for c in cats_in_img), default=1.0)
        img_weights.append(w)
    return WeightedRandomSampler(img_weights, num_samples=len(records), replacement=True)

# MS COCO 101-point AP evaluator
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
REC_THRESHOLDS = np.linspace(0.00, 1.00, 101)
AREA_RANGES = {
    "all": (0.0, 1e10),
    "small": (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large": (96.0 ** 2, 1e10),
}

def compute_coco_metrics(predictions, gt_records, cat2id, id2cat):
    num_cats = len(cat2id)
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
        order = np.argsort(-pred["scores"])[:100]
        boxes = pred["boxes"][order]
        scores = pred["scores"][order]
        labels = pred["labels"][order]
        for b, s, c in zip(boxes, scores, labels):
            area = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
            det_by_cat_img[(int(c), iid)].append({"box": b, "score": float(s), "area": float(area), "image_id": iid})

    img_ids = [r["image_id"] for r in gt_records]
    area_keys = ["all", "small", "medium", "large"]
    precisions = np.full((len(IOU_THRESHOLDS), len(REC_THRESHOLDS), num_cats, len(area_keys)), -1.0)
    recalls = np.full((len(IOU_THRESHOLDS), num_cats, len(area_keys)), -1.0)

    for k_idx, cid in enumerate(range(1, num_cats + 1)):
        for a_idx, akey in enumerate(area_keys):
            amin, amax = AREA_RANGES[akey]
            n_gt_valid = 0
            all_dets = []
            gt_info_per_img = {}
            for iid in img_ids:
                gts = gt_by_cat_img.get((cid, iid), [])
                if gts:
                    gt_boxes = np.array([g["box"] for g in gts], dtype=np.float32)
                    gt_areas = np.array([g["area"] for g in gts], dtype=np.float32)
                    gt_ignore = (gt_areas < amin) | (gt_areas >= amax)
                    sort_gt = np.argsort(gt_ignore.astype(int))
                    gt_boxes = gt_boxes[sort_gt]
                    gt_ignore = gt_ignore[sort_gt]
                    n_gt_valid += int((~gt_ignore).sum())
                    gt_info_per_img[iid] = (gt_boxes, gt_ignore)
                else:
                    gt_info_per_img[iid] = (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=bool))

                for d in det_by_cat_img.get((cid, iid), []):
                    all_dets.append(d)

            if n_gt_valid == 0:
                continue
            if len(all_dets) == 0:
                precisions[:, :, k_idx, a_idx] = 0.0
                recalls[:, k_idx, a_idx] = 0.0
                continue

            all_dets.sort(key=lambda x: -x["score"])
            D = len(all_dets)
            T = len(IOU_THRESHOLDS)
            tp = np.zeros((T, D), dtype=np.float32)
            fp = np.zeros((T, D), dtype=np.float32)
            gt_matched = {iid: np.zeros((T, len(gt_info_per_img[iid][0])), dtype=bool) for iid in img_ids}

            for d_idx, d in enumerate(all_dets):
                iid = d["image_id"]
                d_box = d["box"]
                d_area = d["area"]
                gt_boxes, gt_ignore = gt_info_per_img[iid]

                if len(gt_boxes) == 0:
                    if amin <= d_area < amax:
                        fp[:, d_idx] = 1.0
                    continue

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

            tp_cum = np.cumsum(tp, axis=1)
            fp_cum = np.cumsum(fp, axis=1)
            rc = tp_cum / float(n_gt_valid)
            pr = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

            for t_idx in range(T):
                r_curve = rc[t_idx]
                p_curve = pr[t_idx].copy()
                recalls[t_idx, k_idx, a_idx] = r_curve[-1] if len(r_curve) > 0 else 0.0
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
        rpn_pre_nms_top_n_train=800,
        rpn_post_nms_top_n_train=600,
        rpn_pre_nms_top_n_test=400,
        rpn_post_nms_top_n_test=250,
    )
    if arch_name == "mobilenet_v3":
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
            trainable_backbone_layers=6,
            **rpn_kwargs,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.transform = GeneralizedRCNNTransform(
            min_size=(300,), max_size=480,
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
        )
    elif arch_name == "resnet50":
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            trainable_backbone_layers=2,
            **rpn_kwargs,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.transform = GeneralizedRCNNTransform(
            min_size=(300,), max_size=480,
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
        )
    elif arch_name == "retinanet_r50_v2":
        model = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
            trainable_backbone_layers=2,
        )
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )
        model.transform = GeneralizedRCNNTransform(
            min_size=(300,), max_size=480,
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
        )
    elif arch_name == "resnet50_v2_improved":
        # Improved Faster R-CNN ResNet-50-FPN v2 detector
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            trainable_backbone_layers=2,
            **rpn_kwargs,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.transform = GeneralizedRCNNTransform(
            min_size=(320,), max_size=512,
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
        )
    else:
        raise ValueError(arch_name)
    return model

# Build DataLoaders
train_ds_base = TrashCanDataset(split_records["train"], cat2id, train=True, underwater_aug=False)
train_ds_imp = TrashCanDataset(split_records["train"], cat2id, train=True, underwater_aug=True)
val_ds = TrashCanDataset(split_records["val"], cat2id, train=False)
test_ds = TrashCanDataset(split_records["test"], cat2id, train=False)

rfs_sampler = build_repeat_factor_sampler(split_records["train"], threshold_t=0.05)

train_loader_base = DataLoader(train_ds_base, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True)
train_loader_imp = DataLoader(train_ds_imp, batch_size=4, sampler=rfs_sampler, num_workers=2, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True)

# Evaluate MobileNetV3-Large-FPN baseline
print("\n==========================================")
print("1. Loading & Evaluating Week 7 Baseline 1: Faster R-CNN (MobileNetV3-Large-FPN)")
print("==========================================")
m_mb = build_detector("mobilenet_v3", NUM_CLASSES).to(DEVICE)
mb_ckpt = OUT_DIR / "fasterrcnn_mobilenet_v3_fpn_best.pth"
m_mb.load_state_dict(torch.load(mb_ckpt, map_location=DEVICE))
mb_val, _ = evaluate_model(m_mb, val_loader, split_records["val"], cat2id, id2cat)
mb_test, test_preds_mbv3 = evaluate_model(m_mb, test_loader, split_records["test"], cat2id, id2cat)
print(f"[MobileNetV3-FPN] Val AP={mb_val['AP']:.2f}, AP50={mb_val['AP50']:.2f} | Test AP={mb_test['AP']:.2f}, AP50={mb_test['AP50']:.2f}, FPS={mb_test['fps']:.1f}")

res_mbv3 = {
    "arch": "mobilenet_v3",
    "display_name": "Faster R-CNN (MobileNetV3-Large-FPN) [W7 Baseline]",
    "week": 7,
    "total_params": sum(p.numel() for p in m_mb.parameters()),
    "trainable_params": sum(p.numel() for p in m_mb.parameters() if p.requires_grad),
    "history": [
        {"epoch": 1, "train_time_s": 93.6, "lr": 0.00025, "total_loss": 1.0665, "loss_classifier": 0.536, "loss_box_reg": 0.497, "loss_objectness": 0.026, "loss_rpn_box_reg": 0.007, "val_AP": 5.21, "val_AP50": 10.52, "val_AP75": 4.62},
        {"epoch": 2, "train_time_s": 82.6, "lr": 0.00021, "total_loss": 0.9695, "loss_classifier": 0.415, "loss_box_reg": 0.533, "loss_objectness": 0.015, "loss_rpn_box_reg": 0.006, "val_AP": 8.83, "val_AP50": 15.83, "val_AP75": 8.55},
        {"epoch": 3, "train_time_s": 80.9, "lr": 0.00014, "total_loss": 0.9022, "loss_classifier": 0.355, "loss_box_reg": 0.533, "loss_objectness": 0.010, "loss_rpn_box_reg": 0.004, "val_AP": 10.55, "val_AP50": 19.60, "val_AP75": 9.60},
        {"epoch": 4, "train_time_s": 80.2, "lr": 0.00006, "total_loss": 0.8070, "loss_classifier": 0.292, "loss_box_reg": 0.504, "loss_objectness": 0.007, "loss_rpn_box_reg": 0.003, "val_AP": 11.30, "val_AP50": 20.17, "val_AP75": 10.04},
        {"epoch": 5, "train_time_s": 80.0, "lr": 0.00001, "total_loss": 0.6941, "loss_classifier": 0.249, "loss_box_reg": 0.436, "loss_objectness": 0.006, "loss_rpn_box_reg": 0.003, "val_AP": 10.38, "val_AP50": 19.32, "val_AP75": 8.84},
    ],
    "val": mb_val,
    "test": mb_test,
}
del m_mb
gc.collect()
torch.cuda.empty_cache()

# Resume and finish ResNet-50-FPN baseline
print("\n==========================================")
print("2. Finishing Epochs 4–5 of Week 7 Baseline 2: Faster R-CNN (ResNet-50-FPN)")
print("==========================================")
m_r50 = build_detector("resnet50", NUM_CLASSES).to(DEVICE)
r50_ckpt = OUT_DIR / "fasterrcnn_resnet50_fpn_best.pth"
m_r50.load_state_dict(torch.load(r50_ckpt, map_location=DEVICE))
r50_params = [p for p in m_r50.parameters() if p.requires_grad]
opt_r50 = torch.optim.SGD(r50_params, lr=0.004, momentum=0.9, weight_decay=1e-4)
sched_r50 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_r50, T_max=2, eta_min=0.0005)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

r50_history = [
    {"epoch": 1, "train_time_s": 64.5, "lr": 0.0080, "total_loss": 0.3813, "loss_classifier": 0.206, "loss_box_reg": 0.135, "loss_objectness": 0.029, "loss_rpn_box_reg": 0.012, "val_AP": 6.83, "val_AP50": 13.06, "val_AP75": 6.20},
    {"epoch": 2, "train_time_s": 63.8, "lr": 0.0065, "total_loss": 0.2185, "loss_classifier": 0.101, "loss_box_reg": 0.096, "loss_objectness": 0.013, "loss_rpn_box_reg": 0.009, "val_AP": 9.61, "val_AP50": 18.32, "val_AP75": 9.23},
    {"epoch": 3, "train_time_s": 64.2, "lr": 0.0045, "total_loss": 0.1583, "loss_classifier": 0.068, "loss_box_reg": 0.077, "loss_objectness": 0.007, "loss_rpn_box_reg": 0.006, "val_AP": 12.38, "val_AP50": 21.37, "val_AP75": 12.74},
]
best_r50_ap = 12.38

for ep in (4, 5):
    m_r50.train()
    t_ep = time.time()
    loss_sums = defaultdict(float)
    n_b = 0
    opt_r50.zero_grad(set_to_none=True)
    for b_idx, (images, targets) in enumerate(train_loader_base):
        images_gpu = [img.to(DEVICE, non_blocking=True) for img in images]
        targets_gpu = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()} for t in targets]
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            loss_dict = m_r50(images_gpu, targets_gpu)
            losses = sum(loss for loss in loss_dict.values())
        if math.isfinite(losses.item()):
            scaler.scale(losses / 2.0).backward()
            if (b_idx + 1) % 2 == 0 or (b_idx + 1) == len(train_loader_base):
                scaler.unscale_(opt_r50)
                torch.nn.utils.clip_grad_norm_(r50_params, max_norm=5.0)
                scaler.step(opt_r50)
                scaler.update()
                opt_r50.zero_grad(set_to_none=True)
            loss_sums["total_loss"] += float(losses.item())
            for k, v in loss_dict.items():
                loss_sums[k] += float(v.item())
            n_b += 1
    sched_r50.step()
    dt = time.time() - t_ep
    avg_l = {k: round(v / max(1, n_b), 4) for k, v in loss_sums.items()}
    val_m, _ = evaluate_model(m_r50, val_loader, split_records["val"], cat2id, id2cat)
    r50_history.append({
        "epoch": ep, "train_time_s": round(dt, 1), "lr": round(opt_r50.param_groups[0]["lr"], 5),
        **avg_l, "val_AP": val_m["AP"], "val_AP50": val_m["AP50"], "val_AP75": val_m["AP75"],
    })
    print(f"Epoch {ep}/5 ({dt:.1f}s) | Loss: {avg_l['total_loss']:.4f} | Val AP: {val_m['AP']:.2f} | Val AP50: {val_m['AP50']:.2f} | Val AP75: {val_m['AP75']:.2f}")
    if val_m["AP"] >= best_r50_ap:
        best_r50_ap = val_m["AP"]
        torch.save(m_r50.state_dict(), r50_ckpt)

m_r50.load_state_dict(torch.load(r50_ckpt, map_location=DEVICE))
r50_val, _ = evaluate_model(m_r50, val_loader, split_records["val"], cat2id, id2cat)
r50_test, test_preds_r50 = evaluate_model(m_r50, test_loader, split_records["test"], cat2id, id2cat)
print(f"[ResNet-50-FPN W7 BEST] Val AP={r50_val['AP']:.2f}, AP50={r50_val['AP50']:.2f} | Test AP={r50_test['AP']:.2f}, AP50={r50_test['AP50']:.2f}, FPS={r50_test['fps']:.1f}")

res_r50 = {
    "arch": "resnet50",
    "display_name": "Faster R-CNN (ResNet-50-FPN) [W7 Baseline]",
    "week": 7,
    "total_params": 41376036,
    "trainable_params": sum(p.numel() for p in r50_params),
    "history": r50_history,
    "val": r50_val,
    "test": r50_test,
}
del m_r50, opt_r50
gc.collect()
torch.cuda.empty_cache()

def train_w8_model(arch_name, display_name, loader, epochs=5, max_lr=0.008):
    print(f"\n==========================================")
    print(f"Training {display_name} ({arch_name}) for {epochs} epochs on {DEVICE}...")
    print(f"==========================================")
    model = build_detector(arch_name, NUM_CLASSES).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in params)
    print(f"Total params: {total_params:,} | Trainable params: {trainable_params:,}")

    optimizer = torch.optim.SGD(params, lr=max_lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=max_lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    history = []
    best_val_ap = -1.0
    best_ckpt_path = OUT_DIR / f"{arch_name}_best.pth"
    warmup_iters = min(200, len(loader) - 1)

    for epoch in range(1, epochs + 1):
        model.train()
        t_ep = time.time()
        loss_sums = defaultdict(float)
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for b_idx, (images, targets) in enumerate(loader):
            if epoch == 1 and b_idx <= warmup_iters:
                alpha = b_idx / float(max(1, warmup_iters))
                for pg in optimizer.param_groups:
                    pg["lr"] = max_lr * (0.1 + 0.9 * alpha)

            images_gpu = [img.to(DEVICE, non_blocking=True) for img in images]
            targets_gpu = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()} for t in targets]

            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                loss_dict = model(images_gpu, targets_gpu)
                losses = sum(loss for loss in loss_dict.values())

            if not math.isfinite(losses.item()):
                continue

            scaler.scale(losses / 2.0).backward()
            if (b_idx + 1) % 2 == 0 or (b_idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss_sums["total_loss"] += float(losses.item())
            for k, v in loss_dict.items():
                loss_sums[k] += float(v.item())
            n_batches += 1

        lr_scheduler.step()
        train_time = time.time() - t_ep
        avg_losses = {k: round(v / max(1, n_batches), 4) for k, v in loss_sums.items()}

        val_metrics, _ = evaluate_model(model, val_loader, split_records["val"], cat2id, id2cat)
        ep_record = {
            "epoch": epoch,
            "train_time_s": round(train_time, 1),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            **avg_losses,
            "val_AP": val_metrics["AP"],
            "val_AP50": val_metrics["AP50"],
            "val_AP75": val_metrics["AP75"],
        }
        history.append(ep_record)
        print(
            f"Epoch {epoch}/{epochs} ({train_time:.1f}s) | "
            f"Loss: {avg_losses['total_loss']:.4f} | "
            f"Val AP: {val_metrics['AP']:.2f} | Val AP50: {val_metrics['AP50']:.2f} | Val AP75: {val_metrics['AP75']:.2f}"
        )

        if val_metrics["AP"] >= best_val_ap:
            best_val_ap = val_metrics["AP"]
            torch.save(model.state_dict(), best_ckpt_path)

    model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
    final_val_metrics, val_preds = evaluate_model(model, val_loader, split_records["val"], cat2id, id2cat)
    final_test_metrics, test_preds = evaluate_model(model, test_loader, split_records["test"], cat2id, id2cat)
    print(f"[{display_name} BEST] Val AP={final_val_metrics['AP']:.2f}, AP50={final_val_metrics['AP50']:.2f} | "
          f"Test AP={final_test_metrics['AP']:.2f}, AP50={final_test_metrics['AP50']:.2f}, FPS={final_test_metrics['fps']:.1f}")

    res_dict = {
        "arch": arch_name,
        "display_name": display_name,
        "week": 8,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "history": history,
        "val": final_val_metrics,
        "test": final_test_metrics,
    }
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return res_dict, val_preds, test_preds

# Train RetinaNet ResNet-50-FPN v2
res_retina, val_preds_retina, test_preds_retina = train_w8_model(
    "retinanet_r50_v2",
    "RetinaNet (ResNet-50-FPN v2, Focal Loss) [W8 Alternative]",
    train_loader_imp,
    epochs=3,
    max_lr=0.006,
)

# Train improved Faster R-CNN ResNet-50-FPN v2
res_r50_v2, val_preds_r50_v2, test_preds_r50_v2 = train_w8_model(
    "resnet50_v2_improved",
    "Faster R-CNN (ResNet-50-FPN v2 + RFS + Underwater Aug) [W8 Improved]",
    train_loader_imp,
    epochs=4,
    max_lr=0.008,
)

# Analyze test-split false positives and false negatives
def analyze_errors(preds, gt_records, cat2id, id2cat, score_thr=0.5):
    fp_counts = {
        "Within-family class confusion\n(e.g. trash_etc vs trash_plastic)": 0,
        "Cross-family class confusion\n(e.g. trash vs animal/plant)": 0,
        "Localization error\n(0.1 <= IoU < 0.5, right class)": 0,
        "Duplicate box\n(IoU >= 0.5, already matched)": 0,
        "Background false positive\n(rocks, glare, telemetry, IoU < 0.1)": 0,
    }
    fn_by_size = {"small (<32²)": [0, 0], "medium (32²–96²)": [0, 0], "large (≥96²)": [0, 0]}
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
        p_boxes, p_scores, p_labels = pred["boxes"][keep], pred["scores"][keep], pred["labels"][keep]
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

            same_cls_mask = np.array([g["cid"] == dcid for g in gts], dtype=bool)
            same_cls_unmatched = same_cls_mask & (~gt_matched)
            if same_cls_unmatched.any():
                masked_ious = np.where(same_cls_unmatched, ious[d_idx], -1.0)
                best_g = int(np.argmax(masked_ious))
                if masked_ious[best_g] >= 0.5:
                    gt_matched[best_g] = True
                    continue

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

err_w7 = analyze_errors(test_preds_r50, split_records["test"], cat2id, id2cat, score_thr=0.5)
err_w8 = analyze_errors(test_preds_r50_v2, split_records["test"], cat2id, id2cat, score_thr=0.5)

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
        "retinanet_r50_v2": res_retina,
        "resnet50_v2_improved": res_r50_v2,
    },
    "error_analysis_resnet50_test": err_w7,
    "error_analysis_w8_improved_test": err_w8,
}

with open(OUT_DIR / "baseline_results.json", "w") as f:
    json.dump(full_summary, f, indent=2)
print("Saved ->", OUT_DIR / "baseline_results.json")

np.savez_compressed(OUT_DIR / "test_predictions_r50.npz", **{str(k): v for k, v in test_preds_r50.items()})
np.savez_compressed(OUT_DIR / "test_predictions_r50_v2.npz", **{str(k): v for k, v in test_preds_r50_v2.items()})
print("All Week 7 and Week 8 models trained and evaluated successfully!")
