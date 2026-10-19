import os
import sys
import time
import json
import numpy as np
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.detector import DetectorService, VOC_CLASSES

def compute_iou(box1, box2):
    """
    box1, box2: [xmin, ymin, xmax, ymax]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def voc_ap(rec, prec, use_07_metric=True):
    """
    Compute VOC AP given precision and recall.
    use_07_metric: True for 11-point interpolation (VOC 2007 standard)
    """
    if use_07_metric:
        # 11-point interpolated AP
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # Continuous AP
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def run_evaluation(
    num_images=1000,
    conf_threshold=0.15,
    iou_thresh=0.5,
    use_pyramid=False,
    pyramid_type="both",
    pyramid_scales=None,
    output_filename="evaluation_results.json"
):
    pyr_str = f"with Image Pyramid ({pyramid_type})" if use_pyramid else "Single Scale"
    print(f"=== Starting OverFeat Model Evaluation ({pyr_str}) ===")
    anno_dir = ROOT_DIR / "data" / "VOCdevkit" / "VOC2007" / "Annotations"
    img_dir = ROOT_DIR / "data" / "VOCdevkit" / "VOC2007" / "JPEGImages"
    val_file = ROOT_DIR / "data" / "VOCdevkit" / "VOC2007" / "ImageSets" / "Main" / "val.txt"

    with open(val_file) as f:
        val_ids = [line.strip() for line in f if line.strip()]

    if num_images is not None and num_images < len(val_ids):
        val_ids = val_ids[:num_images]

    print(f"Evaluating on {len(val_ids)} validation images...")

    detector = DetectorService()
    target_classes = VOC_CLASSES[1:]  # exclude background

    # Ground truth structures: class -> image_id -> list of {'box': ..., 'difficult': bool, 'det': bool}
    ground_truths = {cls: defaultdict(list) for cls in target_classes}
    npos_per_class = defaultdict(int)

    # Predictions structure: class -> list of {'image_id': ..., 'confidence': ..., 'box': ...}
    all_detections = {cls: [] for cls in target_classes}

    latencies = []
    size_breakdown_gt = {"small": 0, "medium": 0, "large": 0}
    size_breakdown_detected = {"small": 0, "medium": 0, "large": 0}

    # 1. Load Ground Truths
    for vid in val_ids:
        xml_path = anno_dir / f"{vid}.xml"
        if not xml_path.exists():
            continue
        tree = ET.parse(xml_path)
        for obj in tree.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in target_classes:
                continue
            bnd = obj.find("bndbox")
            box = [
                float(bnd.find("xmin").text),
                float(bnd.find("ymin").text),
                float(bnd.find("xmax").text),
                float(bnd.find("ymax").text),
            ]
            difficult = int(obj.find("difficult").text) if obj.find("difficult") is not None else 0
            
            ground_truths[cls_name][vid].append({
                "box": box,
                "difficult": difficult,
                "detected": False
            })
            if not difficult:
                npos_per_class[cls_name] += 1
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area < 32 * 32:
                    size_breakdown_gt["small"] += 1
                elif area < 96 * 96:
                    size_breakdown_gt["medium"] += 1
                else:
                    size_breakdown_gt["large"] += 1

    # 2. Run Inference
    print(f"Running inference across images (Pyramid: {use_pyramid}, Type: {pyramid_type})...")
    start_eval_time = time.time()
    for idx, vid in enumerate(val_ids):
        img_path = img_dir / f"{vid}.jpg"
        img = Image.open(img_path).convert("RGB")
        
        t0 = time.time()
        pred = detector.predict(
            img,
            confidence_threshold=conf_threshold,
            iou_threshold=0.45,
            use_pyramid=use_pyramid,
            pyramid_type=pyramid_type,
            pyramid_scales=pyramid_scales
        )
        latencies.append((time.time() - t0) * 1000.0)

        for det in pred["detections"]:
            cls_name = det["class_name"]
            if cls_name in all_detections:
                all_detections[cls_name].append({
                    "image_id": vid,
                    "confidence": det["confidence"],
                    "box": det["box"]
                })

    total_eval_time = time.time() - start_eval_time
    fps = len(val_ids) / total_eval_time

    # 3. Compute Per-Class AP, Precision, Recall, and Error Types
    class_metrics = {}
    error_analysis = {
        "true_positives": 0,
        "false_positives_localization": 0,
        "false_positives_confusion": 0,
        "false_positives_background": 0,
        "iou_scores": []
    }

    aps = []
    print("\n--- Per-Class Evaluation (VOC 2007 11-point AP@0.5) ---")

    for cls in target_classes:
        npos = npos_per_class[cls]
        dets = all_detections[cls]

        if len(dets) == 0:
            class_metrics[cls] = {
                "ap": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "n_gt": npos,
                "n_pred": 0
            }
            aps.append(0.0)
            continue

        # Sort detections by confidence descending
        dets = sorted(dets, key=lambda x: x["confidence"], reverse=True)
        nd = len(dets)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        # Deep copy GT detection flags for this class
        gt_class = {
            vid: [dict(obj) for obj in obj_list]
            for vid, obj_list in ground_truths[cls].items()
        }

        for d_idx, d in enumerate(dets):
            vid = d["image_id"]
            pred_box = d["box"]
            gts = gt_class.get(vid, [])

            ovmax = -1.0
            jmax = -1

            if len(gts) > 0:
                for j, gt in enumerate(gts):
                    ov = compute_iou(pred_box, gt["box"])
                    if ov > ovmax:
                        ovmax = ov
                        jmax = j

            if ovmax >= iou_thresh:
                if not gts[jmax]["difficult"]:
                    if not gts[jmax]["detected"]:
                        tp[d_idx] = 1.0
                        gts[jmax]["detected"] = True
                        error_analysis["true_positives"] += 1
                        error_analysis["iou_scores"].append(ovmax)
                        
                        area = (gts[jmax]["box"][2] - gts[jmax]["box"][0]) * (gts[jmax]["box"][3] - gts[jmax]["box"][1])
                        if area < 32 * 32:
                            size_breakdown_detected["small"] += 1
                        elif area < 96 * 96:
                            size_breakdown_detected["medium"] += 1
                        else:
                            size_breakdown_detected["large"] += 1
                    else:
                        fp[d_idx] = 1.0  # duplicate detection
            else:
                fp[d_idx] = 1.0
                if ovmax >= 0.1:
                    error_analysis["false_positives_localization"] += 1
                else:
                    error_analysis["false_positives_background"] += 1

        # Compute precision and recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        rec = cum_tp / float(max(1, npos))
        prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)

        ap = voc_ap(rec, prec, use_07_metric=True)
        aps.append(ap)

        final_rec = float(rec[-1]) if len(rec) > 0 else 0.0
        final_prec = float(prec[-1]) if len(prec) > 0 else 0.0
        f1 = (2 * final_prec * final_rec / (final_prec + final_rec)) if (final_prec + final_rec) > 0 else 0.0

        class_metrics[cls] = {
            "ap": round(float(ap), 4),
            "precision": round(final_prec, 4),
            "recall": round(final_rec, 4),
            "f1": round(f1, 4),
            "n_gt": npos,
            "n_pred": nd,
            "tp_count": int(np.sum(tp)),
            "fp_count": int(np.sum(fp))
        }

        print(f"  {cls:<12} | AP: {ap*100:5.2f}% | Precision: {final_prec*100:5.2f}% | Recall: {final_rec*100:5.2f}% | GT: {npos:<4} | Pred: {nd:<4}")

    mAP = float(np.mean(aps))
    print(f"\n==========================================")
    print(f"  Mean Average Precision (mAP@0.5): {mAP*100:5.2f}%")
    print(f"==========================================")

    # 4. Model Architecture & Complexity Profiling
    total_params = sum(p.numel() for p in detector.model.parameters())
    backbone_params = sum(p.numel() for p in detector.model.backbone.parameters())
    cls_head_params = sum(p.numel() for p in detector.model.classifier_head.parameters())
    reg_head_params = sum(p.numel() for p in detector.model.Regression_bbx_head.parameters())
    model_size_mb = os.path.getsize(str(ROOT_DIR / "model_weight.pth")) / (1024 * 1024)

    mean_tp_iou = float(np.mean(error_analysis["iou_scores"])) if error_analysis["iou_scores"] else 0.0

    results = {
        "mAP_05": round(mAP, 4),
        "mean_tp_iou": round(mean_tp_iou, 4),
        "total_eval_images": len(val_ids),
        "total_eval_time_sec": round(total_eval_time, 2),
        "throughput_fps": round(fps, 1),
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 2),
            "median": round(float(np.median(latencies)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "min": round(float(np.min(latencies)), 2),
            "max": round(float(np.max(latencies)), 2),
        },
        "model_complexity": {
            "total_parameters": total_params,
            "backbone_parameters": backbone_params,
            "classifier_head_parameters": cls_head_params,
            "regression_head_parameters": reg_head_params,
            "checkpoint_size_mb": round(model_size_mb, 2),
            "device": str(detector.device),
        },
        "error_distribution": {
            "true_positives": error_analysis["true_positives"],
            "false_positives_localization": error_analysis["false_positives_localization"],
            "false_positives_background": error_analysis["false_positives_background"],
        },
        "size_recall": {
            "small": {
                "gt_count": size_breakdown_gt["small"],
                "detected": size_breakdown_detected["small"],
                "recall": round(size_breakdown_detected["small"] / max(1, size_breakdown_gt["small"]), 4)
            },
            "medium": {
                "gt_count": size_breakdown_gt["medium"],
                "detected": size_breakdown_detected["medium"],
                "recall": round(size_breakdown_detected["medium"] / max(1, size_breakdown_gt["medium"]), 4)
            },
            "large": {
                "gt_count": size_breakdown_gt["large"],
                "detected": size_breakdown_detected["large"],
                "recall": round(size_breakdown_detected["large"] / max(1, size_breakdown_gt["large"]), 4)
            }
        },
        "per_class": class_metrics,
        "pyramid_config": {
            "use_pyramid": use_pyramid,
            "pyramid_type": pyramid_type,
            "pyramid_scales": pyramid_scales,
        },
    }

    out_json = ROOT_DIR / output_filename
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation results saved to {out_json}")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate OverFeat Model with optional Image Pyramid")
    parser.add_argument("--num-images", type=int, default=500, help="Number of validation images to evaluate")
    parser.add_argument("--conf-threshold", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.50, help="IoU threshold for evaluation AP")
    parser.add_argument("--pyramid", action="store_true", help="Enable multi-scale Image Pyramid (Up + Down scale)")
    parser.add_argument("--pyramid-type", type=str, default="both", choices=["both", "up", "down", "none"], help="Pyramid scale direction")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output JSON filename")
    args = parser.parse_args()

    run_evaluation(
        num_images=args.num_images,
        conf_threshold=args.conf_threshold,
        iou_thresh=args.iou_threshold,
        use_pyramid=args.pyramid,
        pyramid_type=args.pyramid_type,
        output_filename=args.output
    )
