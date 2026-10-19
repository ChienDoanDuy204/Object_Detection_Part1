import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.ops import batched_nms

# Import OverFeatDetector
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.overfeat_model import OverFeatDetector

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# Curated high-contrast color palette for visual overlay
CLASS_COLORS = {
    "aeroplane": "#38bdf8",    # sky blue
    "bicycle": "#a855f7",      # purple
    "bird": "#f43f5e",         # rose
    "boat": "#06b6d4",         # cyan
    "bottle": "#14b8a6",       # teal
    "bus": "#eab308",          # yellow
    "car": "#f97316",          # orange
    "cat": "#ec4899",          # pink
    "chair": "#8b5cf6",        # violet
    "cow": "#84cc16",          # lime
    "diningtable": "#64748b",  # slate
    "dog": "#ef4444",          # red
    "horse": "#d97706",        # amber
    "motorbike": "#6366f1",    # indigo
    "person": "#10b981",       # emerald
    "pottedplant": "#22c55e",   # green
    "sheep": "#0ea5e9",        # light blue
    "sofa": "#a1a1aa",         # zinc
    "train": "#f59e0b",        # amber
    "tvmonitor": "#8b5cf6",    # violet
}

class DetectorService:
    def __init__(self, weight_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        if weight_path is None:
            weight_path = str(ROOT_DIR / "model_weight.pth")
            
        self.num_classes = len(VOC_CLASSES)
        self.model = OverFeatDetector(num_classes=self.num_classes).to(self.device)
        
        if os.path.exists(weight_path):
            state = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            print(f"Loaded weights from {weight_path} onto {self.device}")
        else:
            print(f"Warning: weights not found at {weight_path}. Running with random weights.")
            self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    def get_pyramid_scales(
        self,
        orig_w: int,
        orig_h: int,
        pyramid_type: str = "both",
        scale_step: float = 1.414,
        max_scale: float = 2.0,
        min_dim: int = 224,
        max_dim: int = 960
    ) -> List[float]:
        """
        Calculates valid pyramid scale factors according to pyramid_type ('both', 'up', 'down', 'none').
        Ensures all scaled dimensions respect min_dim (>= 224x224 for OverFeat window) and max_dim.
        """
        if pyramid_type == "none":
            return [1.0]

        return OverFeatDetector.get_pyramid_scales(
            orig_h=orig_h,
            orig_w=orig_w,
            pyramid_type=pyramid_type,
            scale_factor=scale_step,
            max_scale=max_scale,
            min_size=(min_dim, min_dim),
            max_size=(max_dim, max_dim)
        )

    def preprocess_image(self, img: Image.Image, max_dim: int = 640) -> Tuple[torch.Tensor, float, float, int, int]:
        """
        Standard single-scale preprocessing.
        Resizes the image so that minimum dimension >= 224 and maximum dimension <= max_dim,
        preserving aspect ratio. Returns tensor, scale_x, scale_y, orig_w, orig_h.
        """
        orig_w, orig_h = img.size

        # OverFeat sliding window needs at least 224 in both dims
        scale = 1.0
        min_dim = min(orig_w, orig_h)
        if min_dim < 224:
            scale = 224.0 / min_dim
        
        if max(orig_w, orig_h) * scale > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            
        target_w = max(224, int(round(orig_w * scale)))
        target_h = max(224, int(round(orig_h * scale)))
        
        # Round dimensions up to multiples of 32 for optimal feature alignment
        target_w = int(np.ceil(target_w / 32.0) * 32)
        target_h = int(np.ceil(target_h / 32.0) * 32)

        # Recalculate effective scale factors
        scale_x = target_w / float(orig_w)
        scale_y = target_h / float(orig_h)

        resized = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        tensor = self.transform(resized).unsqueeze(0).to(self.device)
        return tensor, scale_x, scale_y, orig_w, orig_h

    def preprocess_scale(
        self,
        img: Image.Image,
        scale: float,
        min_dim: int = 224,
        max_dim: int = 1024
    ) -> Tuple[torch.Tensor, float, float, int, int]:
        """
        Preprocesses image for a specific pyramid scale factor.
        Clamps scaled dimensions to [min_dim, max_dim] and rounds to multiples of 32.
        """
        orig_w, orig_h = img.size

        raw_w = orig_w * scale
        raw_h = orig_h * scale

        # Guarantee at least min_dim for OverFeat sliding window (224x224)
        target_w = max(min_dim, int(round(raw_w)))
        target_h = max(min_dim, int(round(raw_h)))

        # Cap at max_dim
        if max(target_w, target_h) > max_dim:
            down_ratio = max_dim / float(max(target_w, target_h))
            target_w = max(min_dim, int(round(target_w * down_ratio)))
            target_h = max(min_dim, int(round(target_h * down_ratio)))

        # Round to multiples of 32
        target_w = int(np.ceil(target_w / 32.0) * 32)
        target_h = int(np.ceil(target_h / 32.0) * 32)

        scale_x = target_w / float(orig_w)
        scale_y = target_h / float(orig_h)

        resized = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        tensor = self.transform(resized).unsqueeze(0).to(self.device)
        return tensor, scale_x, scale_y, target_w, target_h

    def predict(
        self,
        img: Image.Image,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        allowed_classes: Optional[List[str]] = None,
        max_dim: int = 640,
        use_pyramid: bool = True,
        pyramid_type: str = "both",          # 'both' (up+down), 'up', 'down', 'none'
        pyramid_scales: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Runs OverFeat object detection with optional multi-scale Image Pyramid (Up + Down scale).
        
        When use_pyramid=True:
          - Image is scaled at multiple resolutions (e.g. 0.7x downscale, 1.0x base, 1.4x upscale).
          - Upscaling enlarges small objects so the 224x224 OverFeat sliding window resolves them.
          - Downscaling shrinks large objects into the receptive field.
          - Candidate bounding boxes from all scales are decoded, mapped back to original coordinates,
            and fused together using global batched NMS.
        """
        start_time = time.time()
        orig_w, orig_h = img.size

        # Determine scales to run
        if not use_pyramid or pyramid_type == "none":
            scales_to_run = [1.0]
        elif pyramid_scales is not None and len(pyramid_scales) > 0:
            scales_to_run = sorted(list(set(float(s) for s in pyramid_scales if float(s) > 0)))
        else:
            pyramid_max_dim = max(1000, max_dim)
            scales_to_run = self.get_pyramid_scales(
                orig_w=orig_w,
                orig_h=orig_h,
                pyramid_type=pyramid_type,
                scale_step=1.414,
                max_scale=2.0,
                min_dim=224,
                max_dim=pyramid_max_dim
            )

        preprocess_time = 0.0
        infer_time = 0.0

        candidate_boxes = []
        candidate_scores = []
        candidate_labels = []
        candidate_scales = []
        scale_stats = {}
        total_anchors_evaluated = 0
        grid_info = []

        for s in scales_to_run:
            t_prep = time.time()
            if len(scales_to_run) == 1 and s == 1.0:
                x, scale_x, scale_y, target_w, target_h = self.preprocess_image(img, max_dim=max_dim)
            else:
                x, scale_x, scale_y, target_w, target_h = self.preprocess_scale(
                    img, s, min_dim=224, max_dim=max(max_dim, 960)
                )
            preprocess_time += (time.time() - t_prep) * 1000.0

            t_inf = time.time()
            with torch.no_grad():
                logits, box_deltas = self.model(x)
            infer_time += (time.time() - t_inf) * 1000.0

            B, num_classes, Oh, Ow = logits.shape
            M = Oh * Ow
            total_anchors_evaluated += M
            grid_info.append({"scale": s, "grid_h": Oh, "grid_w": Ow, "anchors": M})

            # Reshape deltas: (1, 4, Oh, Ow) -> (M, 4)
            box_deltas = box_deltas.permute(0, 2, 3, 1).contiguous().view(M, 4)

            # Anchors for current feature map grid
            anchors = self.model.make_anchors(Oh, Ow, batch_size=1, stride=32, window_size=224).to(self.device)
            anchors = anchors.view(M, 4)

            # Decode deltas to coordinates [xmin, ymin, xmax, ymax] in scaled image space
            pred_boxes = self.model.decode_delta(anchors, box_deltas)

            # Probabilities across all classes: (M, num_classes)
            probs = F.softmax(logits, dim=1).squeeze(0).permute(1, 2, 0).contiguous().view(M, num_classes)

            # Exclude background class (index 0)
            foreground_probs = probs[:, 1:]  # (M, 20)
            max_conf, max_cls_offset = torch.max(foreground_probs, dim=1)
            pred_labels = max_cls_offset + 1  # 1-indexed for VOC_CLASSES

            # Filter by confidence threshold per scale to prune background early
            conf_mask = max_conf >= confidence_threshold
            n_pass = int(conf_mask.sum().item())
            scale_tag = f"{s:.2f}x"
            scale_stats[scale_tag] = n_pass

            if n_pass > 0:
                f_boxes = pred_boxes[conf_mask]
                f_scores = max_conf[conf_mask]
                f_labels = pred_labels[conf_mask]

                # Map coordinates back to the original image coordinate frame
                orig_boxes = torch.empty_like(f_boxes)
                orig_boxes[:, 0] = torch.clamp(f_boxes[:, 0] / scale_x, 0.0, float(orig_w))
                orig_boxes[:, 1] = torch.clamp(f_boxes[:, 1] / scale_y, 0.0, float(orig_h))
                orig_boxes[:, 2] = torch.clamp(f_boxes[:, 2] / scale_x, 0.0, float(orig_w))
                orig_boxes[:, 3] = torch.clamp(f_boxes[:, 3] / scale_y, 0.0, float(orig_h))

                candidate_boxes.append(orig_boxes)
                candidate_scores.append(f_scores)
                candidate_labels.append(f_labels)
                candidate_scales.append(torch.full((n_pass,), s, device=self.device, dtype=torch.float32))

        post_start = time.time()
        detections = []

        if len(candidate_boxes) > 0:
            # Fuse multi-scale candidate detections across all pyramid levels
            all_boxes = torch.cat(candidate_boxes, dim=0)
            all_scores = torch.cat(candidate_scores, dim=0)
            all_labels = torch.cat(candidate_labels, dim=0)
            all_scales = torch.cat(candidate_scales, dim=0)

            # Apply global batched NMS to suppress duplicate overlapping boxes across scales
            keep = batched_nms(all_boxes, all_scores, all_labels, iou_threshold=iou_threshold)
            keep_boxes = all_boxes[keep]
            keep_scores = all_scores[keep]
            keep_labels = all_labels[keep]
            keep_scales = all_scales[keep]

            for i in range(len(keep)):
                cls_id = int(keep_labels[i].item())
                cls_name = VOC_CLASSES[cls_id]

                # Check class filtering
                if allowed_classes is not None and len(allowed_classes) > 0 and cls_name not in allowed_classes:
                    continue

                score = float(keep_scores[i].item())
                box = keep_boxes[i].cpu().tolist()
                scale_val = float(keep_scales[i].item())
                scale_role = "up" if scale_val > 1.05 else ("down" if scale_val < 0.95 else "base")

                xmin = max(0.0, min(float(orig_w), box[0]))
                ymin = max(0.0, min(float(orig_h), box[1]))
                xmax = max(0.0, min(float(orig_w), box[2]))
                ymax = max(0.0, min(float(orig_h), box[3]))

                # Ensure valid box dimensions
                if xmax > xmin and ymax > ymin:
                    width = xmax - xmin
                    height = ymax - ymin
                    area = width * height
                    detections.append({
                        "box": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)],
                        "confidence": round(score, 4),
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "color": CLASS_COLORS.get(cls_name, "#38bdf8"),
                        "area": round(area, 1),
                        "width": round(width, 1),
                        "height": round(height, 1),
                        "scale": round(scale_val, 2),
                        "scale_role": scale_role,
                    })

        post_time = (time.time() - post_start) * 1000.0
        total_time = (time.time() - start_time) * 1000.0

        return {
            "image_size": {"width": orig_w, "height": orig_h},
            "grid_size": {
                "total_anchors": total_anchors_evaluated,
                "pyramid_grids": grid_info
            },
            "num_detections": len(detections),
            "detections": detections,
            "pyramid_info": {
                "enabled": use_pyramid and pyramid_type != "none",
                "pyramid_type": pyramid_type if use_pyramid else "none",
                "scales_used": scales_to_run,
                "num_scales": len(scales_to_run),
                "candidates_per_scale": scale_stats,
            },
            "timings_ms": {
                "preprocess": round(preprocess_time, 2),
                "inference": round(infer_time, 2),
                "postprocess": round(post_time, 2),
                "total": round(total_time, 2),
                "fps": round(1000.0 / max(1.0, total_time), 1),
            },
        }

