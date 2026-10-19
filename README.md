# Object Detection Part 1: OverFeat CNN Detector & Real-Time Studio

An end-to-end implementation and interactive web application for **OverFeat Object Detection** (sliding-window CNN architecture) trained on **PASCAL VOC 2007** (20 classes + background).

---

## 📌 Project Context & Overview

### 1. What This Project Does
This project explores the transition from traditional sliding-window / feature-engineering object detection to **modern CNN-based object detection**, specifically implementing the **OverFeat** architecture:
- **Backbone**: Pre-trained ResNet-18 (`timm.create_model('resnet18.a1_in1k', pretrained=True)`).
- **Fully Convolutional Feature Extractor**: Replaces the standard global average pooling and dense classification layer with:
  1. `AvgPool2d(kernel_size=7, stride=1)`: Creates dense sliding receptive field.
  2. `classifier_head = nn.Conv2d(512, num_classes=21, kernel_size=1, stride=1)`: Predicts class logits for each sliding window region.
  3. `Regression_bbx_head = nn.Conv2d(512, 4, kernel_size=1, stride=1)`: Predicts bounding box offsets `(dx, dy, dw, dh)` for each anchor.
- **Anchor Generation**:
  - Sliding window grid with base receptive field `window_size = 224` and `stride = 32`.
- **Target Matching & Loss Function**:
  - Positive IoU threshold: `>= 0.6`
  - Negative (Background) IoU threshold: `<= 0.3`
  - Ignore region: `0.3 < IoU < 0.6`
  - Multi-task Loss: Classification (`CrossEntropyLoss` with `ignore_index=-1`) + Bounding Box Regression (`SmoothL1Loss` on positive anchors).
  - Mixed Precision Training (`torch.amp.GradScaler`).
- **Multi-Scale Image Pyramid (Up + Down Scale Inference)**:
  - OverFeat uses a fixed $224 \times 224$ receptive field sliding window. In single scale inference, small objects fall below the window resolution while very large objects overflow it.
  - **Up-scale Pyramid ($s > 1.0$)**: Enlarges the input image so small objects (birds, distant pedestrians, bottles) become large enough to match the $224 \times 224$ anchor window.
  - **Down-scale Pyramid ($s < 1.0$)**: Shrinks the input image so large objects (cars, horses, aeroplanes) fit within the receptive field.
  - **Global Fusion**: Candidate bounding boxes and classification scores across all pyramid levels are mapped back to original image space and fused using global `batched_nms`.
- **Trained Weights**: Checkpoint saved as [model_weight.pth](file:///home/chiendoan/Code_Module6/Object_Detection_Part1/model_weight.pth) (~44 MB).

---

## 🚀 Interactive Web Application

A full-stack application enabling real-time detection on **Images**, **Uploaded Videos**, and **Live Camera Streams**:

### 1. Backend (`FastAPI`)
- **Location**: [backend/main.py](file:///home/chiendoan/Code_Module6/Object_Detection_Part1/backend/main.py)
- **Port**: `8000`
- **Features**:
  - `GET /api/info`: Model metadata, GPU accelerator info (`NVIDIA P104-100`), class palette.
  - `GET /api/samples`: VOC 2007 preset test images (Bird, Car, Person, Dog, Bicycle, Aeroplane).
  - `POST /api/predict/image`: Image file upload with criteria filters (`confidence_threshold`, `iou_threshold`, `allowed_classes`).
  - `POST /api/predict/frame`: Optimized base64 image endpoint for fast frame streaming.
  - `POST /api/predict/video`: Video upload with server-side frame sampling at user-defined `target_fps` and optional annotated video generation.
  - `WebSocket /ws/stream`: Bi-directional low-latency WebSocket endpoint for real-time camera streaming with dynamic FPS control.

### 2. Frontend (`Next.js`)
- **Location**: [frontend/](file:///home/chiendoan/Code_Module6/Object_Detection_Part1/frontend/)
- **Port**: `3000`
- **Design System**: Vanilla CSS design tokens with sleek dark mode, glassmorphism, responsive canvas scaling, and micro-interactions.
- **Three Operation Modes**:
  1. 📷 **Image Analysis**:
     - Upload custom images or click any of 7 preset VOC samples.
     - Interactive high-DPI canvas overlay with glowing bounding boxes, class pills, and confidence percentages.
     - Detections inspection panel with hover synchronization.
     - Download annotated image and export JSON detections.
  2. 🎥 **Video Stream**:
     - Upload video files (MP4, WebM).
     - Timeline scrub bar, play/pause controls.
     - **FPS Controller Slider (1 to 30 FPS)**: Directly throttles the frame sampling rate sent to the model during playback.
  3. 📹 **Camera Live Stream**:
     - Live webcam stream with camera selector.
     - **FPS Controller Slider (1 to 30 FPS)**: Regulates the camera transmission interval (`1000 / fps` ms).
     - Real-time HUD showing Camera FPS, Inference Latency, Round-Trip Time, and Object Counts.
     - Snapshot capture button.
- **Filtering Criteria**:
  - Confidence Threshold Slider (5% to 95%).
  - IoU (NMS) Overlap Slider (0.10 to 0.90).
  - Category and Class Filter Chips for all 20 VOC classes.
  - Display toggles: Labels, Confidence %, Box Dimensions, Anchor Grid Guide.

---

## 🛠️ How to Run

### Quick Start (One Command)
Run the automated launcher script from the root directory:
```bash
./run.sh
```
This launches:
- **FastAPI backend** on `http://localhost:8000` (API docs at `http://localhost:8000/docs`)
- **Next.js frontend** on `http://localhost:3000`

### Manual Start

#### Backend:
```bash
/home/chiendoan/miniconda3/envs/study_env/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

#### Frontend:
```bash
cd frontend
export PATH="/home/chiendoan/.local/node/bin:$PATH"
npm run dev -- -p 3000
```
