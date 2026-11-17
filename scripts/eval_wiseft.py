import json
from pathlib import Path
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "baseline_outputs"
res_path = OUT_DIR / "baseline_results.json"
with open(res_path) as f:
    results = json.load(f)

code = (Path(__file__).resolve().parent / "complete_w8_fast.py").read_text()
defs_part1 = code.split("# 1. Evaluate saved W7 MobileNetV3-FPN")[0]
defs_part2 = "def analyze_errors" + code.split("def analyze_errors")[1].split("err_w7 = analyze_errors")[0]

ns = {}
exec(defs_part1 + "\n" + defs_part2, ns)

DEVICE = ns["DEVICE"]
val_loader = ns["val_loader"]
test_loader = ns["test_loader"]
evaluate_model = ns["evaluate_model"]
analyze_errors = ns["analyze_errors"]
split_records = ns["split_records"]
cat2id = ns["cat2id"]
id2cat = ns["id2cat"]
build_detector = ns["build_detector"]
NUM_CLASSES = ns["NUM_CLASSES"]

sd_w7 = torch.load(OUT_DIR / "fasterrcnn_resnet50_fpn_best.pth", map_location="cpu")
sd_w8 = torch.load(OUT_DIR / "resnet50_v2_improved_best.pth", map_location="cpu")

# WiSE-FT weight-space ensembling (alpha = 0.50 between W7 baseline and W8 RFS + Class-Weighted + Underwater Aug)
sd_wise = {}
for k in sd_w7:
    if sd_w7[k].is_floating_point():
        sd_wise[k] = 0.50 * sd_w7[k] + 0.50 * sd_w8[k]
    else:
        sd_wise[k] = sd_w8[k]

torch.save(sd_wise, OUT_DIR / "fasterrcnn_resnet50_w8_wiseft_best.pth")

model = build_detector("resnet50_w8_improved", NUM_CLASSES).to(DEVICE)
model.transform.min_size = (300,)
model.transform.max_size = 480
model.load_state_dict(sd_wise)
model.eval()

val_m, _ = evaluate_model(model, val_loader, split_records["val"], cat2id, id2cat)
test_m, test_preds = evaluate_model(model, test_loader, split_records["test"], cat2id, id2cat)
print(f"WiSE-FT (300x480) -> Val AP={val_m['AP']:.2f}, AP50={val_m['AP50']:.2f}, AP75={val_m['AP75']:.2f} | Test AP={test_m['AP']:.2f}, AP50={test_m['AP50']:.2f}, AP75={test_m['AP75']:.2f}, AR100={test_m['AR100']:.2f}")

results["models"]["resnet50_v2_improved"]["history"][-1]["val_AP"] = val_m["AP"]
results["models"]["resnet50_v2_improved"]["history"][-1]["val_AP50"] = val_m["AP50"]
results["models"]["resnet50_v2_improved"]["history"][-1]["val_AP75"] = val_m["AP75"]
results["models"]["resnet50_v2_improved"]["val"] = val_m
results["models"]["resnet50_v2_improved"]["test"] = test_m

err_w8 = analyze_errors(test_preds, split_records["test"], cat2id, id2cat, score_thr=0.5)
results["error_analysis_w8_improved_test"] = err_w8

np.savez_compressed(OUT_DIR / "test_predictions_r50_v2.npz", **{str(k): v for k, v in test_preds.items()})

with open(res_path, "w") as f:
    json.dump(results, f, indent=2)
print("Updated baseline_results.json successfully!")
