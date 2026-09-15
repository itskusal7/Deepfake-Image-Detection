# Deepfake Image Detection: Architecture Documentation

## Executive Summary

ViT Hybrid system for detecting AI-generated/manipulated images. Combines Vision Transformers with Noise Pattern Recognition (NPR), leveraging attention-based fusion for robust predictions.

**Key Metrics (Updated - May 2026):**

- **Model 1 (ViT Hybrid):** 90.9% AUC (on merged 47K dataset), 99.79% AUC (fine-tuning val), 97.5% Accuracy
- **Dataset:** Complete merged dataset (47,130 images: 24,630 fake, 22,500 real)
- **Inference:** 0.3–0.5s on Apple Silicon MPS

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Model 1: ViT Hybrid Detector](#model-1-vit-hybrid-detector)
3. [Training Pipeline](#training-pipeline)
4. [Performance Metrics](#performance-metrics)
5. [Local Setup Guide](#local-setup-guide)
6. [Deployment & API Reference](#deployment--api-reference)

---

## System Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT: Image                               │
│              (File Upload / URL via index.html)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                    [Preprocess]
                 (Resize 224×224, Normalize)
                         │
                  ┌──────▼──────┐
                  │  Model 1    │
                  │  (ViT)      │
                  │  Port 8000  │
                  │  0.3-0.5s   │
                  └──────┬──────┘
                         │
                  raw_prob = σ(logit)
                  prob = 1.0 - raw_prob
                         │
               ┌─────────▼──────────┐
               │  prob > 0.70 → FAKE │
               │  prob < 0.30 → REAL │
               │  else → UNCERTAIN   │
               └─────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │ JSON Response:      │
              │ prediction + conf   │
              │ + Saliency Heatmap  │
              └─────────────────────┘
```

### Project Structure

```
deepfake_image_detection/
├── backend/
│   ├── model1.py                    # ViT Hybrid Model (FastAPI, port 8000)
│   ├── index.html                   # Web UI
│   ├── STARTUP.md                   # Startup guide
│   ├── QUICK_START.md               # 3-step quickstart
│   └── models/
│       └── model1/
│           ├── vit_baseline_best.pt # Baseline checkpoint
│           └── vit_finetuned.pt     # Fine-tuned checkpoint (production)
│
└── model1/
    ├── vit-hybridmodel1.ipynb       # Baseline training notebook
    ├── fine_tune.py                 # Fine-tuning script
    └── final_dataset_A/
        ├── train/{fake,real}/
        ├── val/{fake,real}/
        └── cross_gen_test/
```

---

## Model 1: ViT Hybrid Detector

### Architecture Overview

**File:** `backend/model1.py`  
**Port:** `8000`  
**Device:** MPS (Apple Silicon) → CPU fallback  
**Inference:** 0.3–0.5 seconds/image  
**Checkpoint:** `models/model1/vit_finetuned.pt`

### Component Details

```python
HybridModel
│
├── Branch 1: Vision Transformer (ViT)
│   ├── Model: vit_small_patch16_224 (timm, pretrained=False at inference)
│   ├── Input: RGB [B, 3, 224, 224]
│   ├── Patches: 14×14 grid = 196 patches
│   ├── Output: 384-d feature vector
│   └── vit_fc: Linear(384 → 256)
│
├── Branch 2: Noise Pattern Recognition (NPR) CNN
│   ├── Artifact extraction: |x[:, :, :-2, :-2] - x[:, :, 2:, 2:]|
│   │   (diagonal pixel diff captures generation artifacts)
│   ├── CNN Stack:
│   │   ├── Conv2d(3→32, 3×3) + BatchNorm2d + ReLU
│   │   ├── Conv2d(32→64, 3×3) + BatchNorm2d + ReLU
│   │   ├── Conv2d(64→128, 3×3) + ReLU
│   │   └── AdaptiveAvgPool2d(1×1)
│   └── npr_fc: Linear(128 → 256)
│
├── Attention Fusion
│   ├── feats = stack([vit_feat, npr_feat])  → [B, 2, 256]
│   ├── attn: Linear(256→128) + ReLU + Linear(128→1)
│   ├── weights = softmax(attn(feats))       → [B, 2]
│   └── fused = Σ(weights_i × feat_i)       → [B, 256]
│
└── Classification Head
    ├── Linear(256→128) + ReLU + Dropout(0.3)
    ├── Linear(128→1)
    └── Output: logit (BCEWithLogitsLoss)
```

### Forward Pass (actual code)

```python
def forward(self, x):
    # ViT branch
    vit_feat = self.vit_fc(self.vit(x))                        # [B, 256]

    # NPR branch — diagonal diff artifact map
    npr = torch.abs(x[:, :, :-2, :-2] - x[:, :, 2:, 2:])
    npr_feat = self.npr_fc(self.npr_cnn(npr).view(x.size(0), -1))  # [B, 256]

    # Attention fusion
    feats   = torch.stack([vit_feat, npr_feat], dim=1)          # [B, 2, 256]
    weights = torch.softmax(self.attn(feats).squeeze(-1), dim=1)# [B, 2]
    fused   = (feats * weights.unsqueeze(-1)).sum(dim=1)        # [B, 256]

    return self.head(fused).squeeze(1)                          # [B]
```

### Prediction Logic

```
raw_prob = sigmoid(logit)
prob     = 1.0 - raw_prob       ← label inversion correction

if prob > 0.70  →  FAKE
if prob < 0.30  →  REAL
else            →  UNCERTAIN    ← model not confident
```

> **Why UNCERTAIN?** Baseline trained with label inversion (ImageFolder alphabetical order). The 0.30–0.70 band is genuinely ambiguous. Honest signal — don't force binary.

### Heatmap Generation (Gradient Saliency)

```python
def generate_heatmap(model, image_tensor):
    img = image_tensor.detach().clone().requires_grad_(True)
    score = torch.sigmoid(model(img))
    model.zero_grad()
    score.backward()

    saliency = img.grad[0].abs().mean(dim=0).cpu().numpy()
    saliency  = np.maximum(saliency, 0)
    saliency /= (saliency.max() + 1e-8)
    heatmap   = cv2.GaussianBlur(saliency, (11, 11), 2)
    return heatmap   # → encoded as base64 PNG (COLORMAP_JET)
```

```
Heatmap color guide:
  RED   → high-gradient regions (model sees artifacts here)
  BLUE  → low-influence regions
```

### Data Preprocessing

```
Input Image (any size, any format)
    ↓
PIL.open().convert("RGB")
    ↓
Resize(224×224)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ↓
Tensor: [1, 3, 224, 224] on MPS/CPU
```

---

## Training Pipeline

### Overview: 3-Stage Training

```
Stage 1: Baseline — Phase 1
         ViT frozen, NPR+Fusion+Head trained (291K params)
         Dataset: final_dataset_A (30,000 images)
         LR: 1e-4
              ↓
Stage 2: Baseline — Phase 2
         Last 2 ViT blocks unfrozen (more params)
         Same dataset, LR: 3e-5
         Output: vit_baseline_best.pt
              ↓
Stage 3: Fine-tuning on New Data
         ALL params unfrozen (21.9M), LR: 5e-6
         Dataset: test_images (10,976 diverse new images)
         Input:  vit_baseline_best.pt
         Output: vit_finetuned.pt   ← used by model1.py
```

---

### Stage 1+2: Baseline Training (`vit-hybridmodel1.ipynb`)

**Dataset: `final_dataset_A`**

| Split          | Real   | Fake   | Total  |
| -------------- | ------ | ------ | ------ |
| Train          | 15,000 | 15,000 | 30,000 |
| Val            | 2,500  | 2,500  | 5,000  |
| cross_gen_test | 2,500  | 2,500  | 5,000  |

Image sizes: 256×256 (real), mixed 256/512/1024px (fake — diffusion output)  
Sources: ProGAN, StyleGAN, Diffusion models, VAE (fake) | Human aligned faces (real)

#### Phase 1: Feature Extraction (5 epochs)

ViT backbone **frozen** — only NPR branch, Fusion, Head trained.

```
Trainable params: 291,074   (of ~22M total)
Optimizer:        AdamW(lr=1e-4, weight_decay=1e-4)
Scheduler:        CosineAnnealingLR(T_max=5, eta_min=1e-6)
Loss:             BCEWithLogitsLoss
Batch size:       32
Device:           MPS (Apple Silicon)
```

| Epoch | Val AUC    | Val Acc      |
| ----- | ---------- | ------------ |
| 1     | 0.9670     | 0.9056       |
| 2     | 0.9728     | 0.8848       |
| 3     | 0.9771     | 0.9210       |
| 4     | 0.9784     | 0.9210       |
| **5** | **0.9787** | **0.9228** ★ |

#### Phase 2: Fine-tuning Backbone (5 epochs)

Last 2 ViT blocks **unfrozen**, rest of ViT still frozen.

```
Optimizer:  AdamW(lr=3e-5, weight_decay=1e-4)
Scheduler:  CosineAnnealingLR(T_max=5, eta_min=1e-7)
```

Best checkpoint → `vit_baseline_best.pt` | Best Phase 2 Val AUC: **~0.979+**

#### Training Augmentations

```
Resize(224×224)
RandomHorizontalFlip(p=0.5)
ToTensor()
Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
```

---

### Stage 3: Fine-tuning on New Data (`fine_tune.py`)

**Why:** Baseline trained on single balanced source. Fine-tuning adds diversity and fixes label convention bug.

**New Dataset: `backend/test_images/`**

| Folder                          | Label | Count      |
| ------------------------------- | ----- | ---------- |
| Human Faces/AI-Generated Images | FAKE  | 4,630      |
| ai_generatedimages/jpeg         | FAKE  | 308        |
| ai_generatedimages/png          | FAKE  | 60         |
| ai_images/test/fake             | FAKE  | 389        |
| ai_images/train/fake            | FAKE  | 153        |
| Human Faces/Real Images         | REAL  | 5,000      |
| ai_images/test/real             | REAL  | 110        |
| ai_images/train/real            | REAL  | 326        |
| **Total**                       |       | **10,976** |

Split: 80/20 stratified → **Train: 8,780 | Val: 2,196**

**Config:**

```
Input checkpoint:  vit_baseline_best.pt
Output checkpoint: vit_finetuned.pt
Batch size:        32
Epochs:            8
LR:                5e-6   ← very low to preserve learned weights
Optimizer:         AdamW(lr=5e-6, weight_decay=1e-4)
Scheduler:         CosineAnnealingLR(T_max=8, eta_min=1e-7)
Loss:              BCEWithLogitsLoss
Frozen layers:     NONE — all 21,956,738 params trainable
```

**Label Convention Fix:**  
Baseline used ImageFolder (alphabetical → fake=0 accidentally). Fine-tuning explicitly assigns fake=1, real=0, then eval flips: `probs = 1.0 - sigmoid(logit)`.

**Augmentations:**

```
Resize(224×224)
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.2, contrast=0.2)
ToTensor()
Normalize(ImageNet mean/std)
```

**Fine-tuning Results:**

| Epoch | Avg Loss   | Val AUC    | Val Acc    | Time           |
| ----- | ---------- | ---------- | ---------- | -------------- |
| 1     | 0.1614     | 0.9950     | 0.9567     | 1019s ★        |
| 2     | 0.0552     | 0.9968     | 0.9672     | 570s ★         |
| 3     | 0.0335     | 0.9973     | 0.9690     | 617s ★         |
| 4     | 0.0217     | 0.9976     | 0.9713     | 595s ★         |
| 5     | 0.0138     | 0.9977     | 0.9736     | 621s ★         |
| 6     | 0.0107     | 0.9978     | 0.9745     | 602s ★         |
| **7** | **0.0087** | **0.9979** | **0.9750** | **649s ★**     |
| 8     | 0.0073     | 0.9979     | 0.9750     | 723s (no gain) |

**Best checkpoint: Epoch 7 → `vit_finetuned.pt`**  
`Best Val AUC: 0.9979  |  Val Accuracy: 97.50%`

---

### Accuracy Progression: Baseline → Fine-tuned

```
                    Val AUC    Val Acc    Dataset
                    ───────    ───────    ───────────────────────────
Baseline Phase 1    0.9787     92.28%    final_dataset_A (5K val)
Baseline Phase 2    ~0.979+    ~92%+     final_dataset_A (5K val)
Fine-tuned Epoch 7  0.9979     97.50%    test_images (2,196 val)  ← production
```

---

## Performance Metrics

### On Fine-tuning Val Set (2,196 images)

```
Best Val AUC:      0.9979
Best Val Accuracy: 97.50%
```

### On Complete Merged Dataset (47,130 images — harder, more diverse)

```
VALIDATION (7,069 images):
  Accuracy:    77.03%
  F1 Score:    0.8023
  AUC:         0.9055
  Sensitivity: 97.6%   (3296 TP / 3375 Fakes)  ← catches almost all fakes
  Specificity: 58.2%   (2149 TN / 3694 Reals)
  Confusion:   TN=2149, FP=1545, FN=79, TP=3296

TEST (7,071 images):
  Accuracy:    77.39%
  F1 Score:    0.8050
  AUC:         0.9090
  Sensitivity: 97.8%   (3300 TP / 3376 Fakes)
  Specificity: 58.8%   (2172 TN / 3695 Reals)
  Confusion:   TN=2172, FP=1523, FN=76, TP=3300
```

> Lower accuracy on merged dataset is **expected** — 47K images span more generators and real-face sources than fine-tuning data. AUC stays strong (90.9%); the 77% accuracy reflects a conservative threshold (prefers FAKE when uncertain).

**Key improvements from fine-tuning:**

| Metric  | Baseline | Fine-tuned | Delta |
| ------- | -------- | ---------- | ----- |
| Val AUC | 0.9787   | 0.9979     | +1.9% |
| Val Acc | 92.3%    | 97.5%      | +5.2% |

---

## Local Setup Guide

### Prerequisites

```bash
python >= 3.9
pip install fastapi uvicorn torch torchvision pillow requests opencv-python pydantic timm
```

### Directory Structure Required

```
backend/
├── model1.py
├── index.html
└── models/
    └── model1/
        └── vit_finetuned.pt    ← place trained checkpoint here
```

### Step 1: Place Model Checkpoint

```bash
cd deepfake_image_detection/backend
mkdir -p models/model1
# Copy vit_finetuned.pt into models/model1/
ls models/model1/vit_finetuned.pt   # verify
```

### Step 2: Start Model 1 Server

```bash
cd deepfake_image_detection/backend
python model1.py
```

**Expected startup output:**

```
Model path exists: True
✅ Hybrid model loaded
INFO:     Uvicorn running on http://0.0.0.0:8000
```

> Device auto-selects: MPS on Apple Silicon, falls back to CPU.

### Step 3: Open Web UI

```bash
open http://localhost:8000/index.html
```

> index.html posts to `http://0.0.0.0:8000/predict`. Must start model1.py first.

### Step 4: Test a Prediction

**Via Web UI:**

1. Select "Model 1: ViT Hybrid" (default selected)
2. Upload image file OR paste image URL
3. Click "Analyze Image"
4. See: prediction label, confidence %, saliency heatmap

**Via curl:**

```bash
# Health check
curl http://localhost:8000/

# Upload file
curl -X POST \
  -F "file=@/path/to/image.jpg" \
  http://localhost:8000/predict

# From URL
curl -X POST \
  -F "image_url=https://example.com/face.jpg" \
  http://localhost:8000/predict
```

### API Response Format

```json
{
  "prediction": "FAKE",
  "confidence": 0.8712,
  "raw_prob": 0.1288,
  "heatmap": "<base64-encoded PNG>",
  "interpretation": "Fake probability: 87.1%. Prediction: FAKE. UNCERTAIN = model not confident (0.35–0.65 range)."
}
```

| Field            | Type   | Values                        |
| ---------------- | ------ | ----------------------------- |
| `prediction`     | string | `FAKE` / `REAL` / `UNCERTAIN` |
| `confidence`     | float  | 0.0–1.0 (fake probability)    |
| `raw_prob`       | float  | Raw sigmoid output            |
| `heatmap`        | string | Base64 PNG (JET colormap)     |
| `interpretation` | string | Human-readable explanation    |

### Prediction Thresholds

```
prob = 1.0 - raw_prob

prob > 0.70  →  FAKE       (confident fake)
prob < 0.30  →  REAL       (confident real)
0.30–0.70   →  UNCERTAIN   (model not confident)
```

### Troubleshooting

| Problem                     | Cause                 | Fix                                            |
| --------------------------- | --------------------- | ---------------------------------------------- |
| `Model path exists: False`  | Wrong checkpoint path | Place `vit_finetuned.pt` in `models/model1/`   |
| Port 8000 in use            | Another process       | `kill -9 $(lsof -t -i :8000)`                  |
| `ModuleNotFoundError: timm` | Missing dependency    | `pip install timm`                             |
| Slow inference              | No MPS                | Normal on CPU; MPS gives 0.3s on M-series      |
| All predictions UNCERTAIN   | Checkpoint mismatch   | Verify checkpoint from same training run       |
| CORS error in browser       | Browser security      | Use `http://localhost:8000` not `file://` path |

---

## Deployment & API Reference

### Backend Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  index.html (Browser)                   │
│        Upload / URL → POST /predict → show result       │
├─────────────────────────────────────────────────────────┤
│                     CORS Middleware                     │
├─────────────────────────────────────────────────────────┤
│                     Port 8000                           │
│          ┌──────────────────────────────┐               │
│          │  model1.py  (FastAPI)        │               │
│          │  HybridModel (ViT + NPR)     │               │
│          │  Gradient Saliency Heatmap   │               │
│          └──────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

### API Endpoints

```
GET  /          → health check → {"status": "Server is running"}
POST /predict   → multipart/form-data
    file        → binary image upload
    image_url   → string URL (Form field)
```

### Image Processing Pipeline

```
Input (any format/size)
    ↓
PIL.open().convert("RGB")
    ↓
Resize(224×224) → ToTensor() → Normalize(ImageNet stats)
    ↓
[1, 3, 224, 224] tensor on MPS/CPU
    ↓
HybridModel.forward()
    ↓
sigmoid → flip (1 - raw_prob) → threshold
    ↓
generate_heatmap() → base64 PNG (COLORMAP_JET)
    ↓
JSON response
```

### Performance

```
Hardware: MacBook Pro M-series (MPS)

Model 1 (ViT):
  Single image:    0.32s
  Throughput:      ~55 img/s
  Memory:          ~310 MB (weights: 87 MB, activations: ~156 MB)
```

---

## Quality Assurance

### Failure Modes

| Failure      | Cause              | Mitigation                      |
| ------------ | ------------------ | ------------------------------- |
| High FP rate | Model conservative | Raise FAKE threshold above 0.70 |
| High FN rate | Threshold too high | Lower threshold                 |
| Model drift  | Dataset shift      | Retrain on recent data          |
| Memory OOM   | Large batch        | Use batch size 1 in inference   |
| GPU failure  | Hardware           | MPS → CPU fallback automatic    |

---

requirements:
pip install fastapi uvicorn timm peft opencv-python torch torchvision
pip install numpy matplotlib pydantic requests

## References

**Model:**

- ViT: `vit_small_patch16_224` (timm library)

**Frameworks:**

- PyTorch 2.0+, FastAPI, timm, OpenCV

**Paper:**

- Dosovitskiy et al. (2020) — An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

---

**Document Version:** 2.1  
**Last Updated:** May 2026  
**Status:** Production Ready
