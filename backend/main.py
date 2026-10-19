import os
import io
import time
import json
import base64
import tempfile
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.detector import DetectorService, VOC_CLASSES, CLASS_COLORS

app = FastAPI(
    title="OverFeat Object Detection API",
    description="FastAPI Backend for OverFeat Object Detection on VOC 2007 with Frame/Stream FPS Control",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Detector Service
detector = DetectorService()

# Directory for sample images
SAMPLES_DIR = ROOT_DIR / "data" / "VOCdevkit" / "VOC2007" / "JPEGImages"
OUTPUTS_DIR = ROOT_DIR / "backend" / "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

PRESET_SAMPLES = [
    {"id": "bird", "label": "Bird in Flight", "filename": "000046.jpg", "category": "animal"},
    {"id": "person", "label": "Person & Street", "filename": "000017.jpg", "category": "person"},
    {"id": "car", "label": "Car on Road", "filename": "000007.jpg", "category": "vehicle"},
    {"id": "dog", "label": "Dog Outdoors", "filename": "000036.jpg", "category": "animal"},
    {"id": "bicycle", "label": "Bicycle", "filename": "000016.jpg", "category": "vehicle"},
    {"id": "aeroplane", "label": "Aeroplane Landing", "filename": "000033.jpg", "category": "vehicle"},
    {"id": "bird2", "label": "Bird on Branch", "filename": "006309.jpg", "category": "animal"},
]

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "OverFeat Object Detection API",
        "model": "OverFeat (ResNet-18 Backbone)",
        "device": str(detector.device),
        "dataset": "PASCAL VOC 2007 (20 classes)",
    }

@app.get("/api/info")
def get_info():
    return {
        "model_name": "OverFeat Object Detector",
        "backbone": "ResNet-18 (timm pretrained)",
        "device": str(detector.device),
        "num_classes": len(VOC_CLASSES),
        "classes": VOC_CLASSES[1:],  # exclude background
        "all_classes": VOC_CLASSES,
        "class_colors": CLASS_COLORS,
        "preset_samples": PRESET_SAMPLES,
        "image_pyramid": {
            "supported": True,
            "pyramid_types": ["both", "up", "down", "none"],
            "default_type": "both",
            "default_scales": [0.707, 1.0, 1.414],
            "description": "Multi-scale Image Pyramid with Up + Down scale resolution sampling to detect multi-sized objects."
        }
    }

@app.get("/api/samples")
def get_samples():
    available_samples = []
    for s in PRESET_SAMPLES:
        file_path = SAMPLES_DIR / s["filename"]
        if file_path.exists():
            available_samples.append({
                **s,
                "url": f"/api/sample-image/{s['filename']}"
            })
    return {"samples": available_samples}

@app.get("/api/sample-image/{filename}")
def get_sample_image(filename: str):
    file_path = SAMPLES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(str(file_path), media_type="image/jpeg")

@app.post("/api/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    allowed_classes: Optional[str] = Form(None),
    max_dim: int = Form(640),
    use_pyramid: bool = Form(True),
    pyramid_type: str = Form("both"),
    pyramid_scales: Optional[str] = Form(None),
):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    class_list = None
    if allowed_classes:
        try:
            # Handle comma-separated or JSON list
            if allowed_classes.startswith("["):
                class_list = json.loads(allowed_classes)
            else:
                class_list = [c.strip() for c in allowed_classes.split(",") if c.strip()]
        except Exception:
            class_list = None

    parsed_scales = None
    if pyramid_scales:
        try:
            if isinstance(pyramid_scales, str) and pyramid_scales.strip().startswith("["):
                parsed_scales = [float(x) for x in json.loads(pyramid_scales)]
            elif isinstance(pyramid_scales, str):
                parsed_scales = [float(x.strip()) for x in pyramid_scales.split(",") if x.strip()]
        except Exception:
            parsed_scales = None

    result = detector.predict(
        img=img,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        allowed_classes=class_list,
        max_dim=max_dim,
        use_pyramid=use_pyramid,
        pyramid_type=pyramid_type,
        pyramid_scales=parsed_scales
    )
    result["filename"] = file.filename
    return result

@app.post("/api/predict/frame")
async def predict_frame(payload: dict):
    """
    Optimized endpoint for video/webcam frame processing.
    Expects JSON payload with:
      - frame: base64 encoded image string (data:image/jpeg;base64,...)
      - confidence_threshold: float (optional)
      - iou_threshold: float (optional)
      - allowed_classes: list[str] (optional)
      - max_dim: int (optional, default 480 for streaming speed)
      - use_pyramid: bool (optional, default False for high streaming FPS)
      - pyramid_type: str (optional, default 'both')
      - pyramid_scales: list[float] (optional)
    """
    frame_data = payload.get("frame")
    if not frame_data:
        raise HTTPException(status_code=400, detail="Missing 'frame' in payload")

    if "," in frame_data:
        frame_data = frame_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(frame_data)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 frame: {str(e)}")

    confidence_threshold = float(payload.get("confidence_threshold", 0.25))
    iou_threshold = float(payload.get("iou_threshold", 0.45))
    allowed_classes = payload.get("allowed_classes")
    max_dim = int(payload.get("max_dim", 480))
    use_pyramid = bool(payload.get("use_pyramid", False))
    pyramid_type = str(payload.get("pyramid_type", "both"))
    pyramid_scales = payload.get("pyramid_scales")

    result = detector.predict(
        img=img,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        allowed_classes=allowed_classes,
        max_dim=max_dim,
        use_pyramid=use_pyramid,
        pyramid_type=pyramid_type,
        pyramid_scales=pyramid_scales
    )
    return result

@app.post("/api/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    target_fps: float = Form(5.0),
    confidence_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    allowed_classes: Optional[str] = Form(None),
    annotate: bool = Form(False),
    use_pyramid: bool = Form(False),
    pyramid_type: str = Form("both"),
):
    """
    Upload a video file, process frames at target_fps, and return detection timeline
    or annotated video.
    """
    class_list = None
    if allowed_classes:
        try:
            if allowed_classes.startswith("["):
                class_list = json.loads(allowed_classes)
            else:
                class_list = [c.strip() for c in allowed_classes.split(",") if c.strip()]
        except Exception:
            class_list = None

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_in:
        temp_in.write(await file.read())
        temp_in_path = temp_in.name

    try:
        cap = cv2.VideoCapture(temp_in_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot read uploaded video file")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / orig_fps if orig_fps > 0 else 0

        # Calculate step interval to achieve target_fps
        target_fps = max(0.5, min(30.0, float(target_fps)))
        step = max(1, int(round(orig_fps / target_fps)))
        effective_fps = orig_fps / step

        frame_results = []
        frame_idx = 0
        processed_count = 0

        out_writer = None
        annotated_video_filename = None
        if annotate:
            annotated_video_filename = f"annotated_{int(time.time())}_{file.filename}"
            annotated_path = str(OUTPUTS_DIR / annotated_video_filename)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(annotated_path, fourcc, effective_fps, (orig_w, orig_h))

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                timestamp = round(frame_idx / orig_fps, 2)
                # Convert BGR to RGB PIL Image
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                pred = detector.predict(
                    img=pil_img,
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold,
                    allowed_classes=class_list,
                    max_dim=480,
                    use_pyramid=use_pyramid,
                    pyramid_type=pyramid_type,
                )

                frame_results.append({
                    "frame_index": frame_idx,
                    "timestamp": timestamp,
                    "num_detections": pred["num_detections"],
                    "detections": pred["detections"],
                    "timings_ms": pred["timings_ms"],
                })

                if out_writer is not None:
                    # Draw bounding boxes on frame_bgr
                    for det in pred["detections"]:
                        box = [int(v) for v in det["box"]]
                        color_hex = det.get("color", "#38bdf8").lstrip("#")
                        bgr_color = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0)) # Hex to BGR
                        cv2.rectangle(frame_bgr, (box[0], box[1]), (box[2], box[3]), bgr_color, 2)
                        label = f"{det['class_name']} {int(det['confidence']*100)}%"
                        cv2.putText(frame_bgr, label, (box[0], max(15, box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)
                    out_writer.write(frame_bgr)

                processed_count += 1
                # Cap processing to avoid timeout on very long videos (max 300 sampled frames)
                if processed_count >= 300:
                    break

            frame_idx += 1

        cap.release()
        if out_writer is not None:
            out_writer.release()

        annotated_url = None
        if annotated_video_filename:
            annotated_url = f"/api/outputs/{annotated_video_filename}"

        return {
            "video_info": {
                "filename": file.filename,
                "original_fps": round(orig_fps, 2),
                "target_fps": target_fps,
                "effective_fps": round(effective_fps, 2),
                "total_frames": total_frames,
                "processed_frames": processed_count,
                "duration_seconds": round(duration_sec, 2),
                "dimensions": {"width": orig_w, "height": orig_h},
            },
            "annotated_video_url": annotated_url,
            "timeline": frame_results,
        }
    finally:
        if os.path.exists(temp_in_path):
            os.remove(temp_in_path)

@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Real-time WebSocket endpoint for camera/video streaming.
    Client sends JSON:
      {
        "frame": "data:image/jpeg;base64,...",
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "allowed_classes": ["person", "car"],
        "max_dim": 448
      }
    Server responds with:
      {
        "status": "success",
        "detections": [...],
        "timings_ms": {...},
        "server_time": 1729000000.123
      }
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                frame_data = msg.get("frame")
                if not frame_data:
                    await websocket.send_text(json.dumps({"error": "No frame received"}))
                    continue

                if "," in frame_data:
                    frame_data = frame_data.split(",", 1)[1]

                image_bytes = base64.b64decode(frame_data)
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                conf_thresh = float(msg.get("confidence_threshold", 0.25))
                iou_thresh = float(msg.get("iou_threshold", 0.45))
                allowed_classes = msg.get("allowed_classes")
                max_dim = int(msg.get("max_dim", 448))
                use_pyramid = bool(msg.get("use_pyramid", False))
                pyramid_type = str(msg.get("pyramid_type", "both"))
                pyramid_scales = msg.get("pyramid_scales")

                res = detector.predict(
                    img=img,
                    confidence_threshold=conf_thresh,
                    iou_threshold=iou_thresh,
                    allowed_classes=allowed_classes,
                    max_dim=max_dim,
                    use_pyramid=use_pyramid,
                    pyramid_type=pyramid_type,
                    pyramid_scales=pyramid_scales,
                )

                response_payload = {
                    "status": "success",
                    "num_detections": res["num_detections"],
                    "detections": res["detections"],
                    "pyramid_info": res.get("pyramid_info"),
                    "timings_ms": res["timings_ms"],
                    "client_timestamp": msg.get("client_timestamp"),
                    "server_time": time.time(),
                }
                await websocket.send_text(json.dumps(response_payload))

            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))

    except WebSocketDisconnect:
        pass

# Mount static outputs folder if needed
app.mount("/api/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
