# 🚀 Dual-Model Deepfake Detection System

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface (index.html)               │
│              - Model selection (radio buttons)              │
│              - File upload / URL input                      │
│              - Result display with heatmap                  │
└─────────────────────────────────────────────────────────────┘
         ↙                                          ↘
    Port 8000                                   Port 8081
    ✅ Model 1                                  ✅ Model 2
    ViT Hybrid                                  DNF-Based
    (Faster)                                    (More Accurate)
```

---

## Model Descriptions

### **Model 1: ViT Hybrid (Port 8000)**

- **Architecture:** Vision Transformer + NPR (Normalized Pixel Residuals)
- **Approach:** Feature extraction from multiple branches + attention fusion
- **Speed:** ~0.3-0.5 seconds per image
- **Best for:** Real-time detection, general deepfakes
- **File:** `main.py`
- **Checkpoint:** `models/vit_baseline_best.pt`

### **Model 2: DNF-Based (Port 8081)**

- **Architecture:** DNF (Diffusion-based Features) + NPR
- **Approach:** Extracts features from diffusion model + pixel artifacts
- **Speed:** ~2-5 seconds per image (depends on GPU)
- **Best for:** Sophisticated deepfakes, high accuracy needed
- **File:** `model2.py`
- **Checkpoint:** `models/detector_final.pt`
- **Dependencies:** Diffusion models (AutoencoderKL, UNet2D, LCMScheduler)

---

## Installation

### Prerequisites

```bash
# Required packages
pip install fastapi uvicorn torch torchvision pillow requests opencv-python
pip install timm  # For Model 1
pip install diffusers  # For Model 2
```

### Download Model Checkpoints

```bash
cd /Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend

# Model 1 checkpoint
mkdir -p models
# Place vit_baseline_best.pt in models/

# Model 2 checkpoint
# Place detector_final.pt in models/
```

---

## Running the System

### Terminal 1: Start Model 1 (Port 8000)

```bash
cd /Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend
python main.py
```

**Expected output:**

```
✅ Hybrid model loaded
───────────────────────
🚀 Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2: Start Model 2 (Port 8081)

```bash
cd /Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend
python model2.py
```

**Expected output:**

```
Device: mps
Loading Detector model...
✅ Model loaded
Loading diffusion models (VAE + UNet)...
✅ Diffusion models ready

============================================================
🚀 Starting Model 2 (DNF Detector) on port 8081
============================================================
Uvicorn running on http://0.0.0.0:8081
```

### Terminal 3: Open Web Interface

```bash
# Option 1: Open directly
open http://localhost:8000/index.html

# Option 2: Use Python server
cd backend && python -m http.server 8888
# Then visit http://localhost:8888/index.html
```

---

## API Endpoints

### Model 1 (Port 8000)

```bash
# Health check
curl http://localhost:8000/

# Prediction (file)
curl -X POST -F "file=@image.jpg" http://localhost:8000/predict

# Prediction (URL)
curl -X POST -F "image_url=https://example.com/image.jpg" http://localhost:8000/predict
```

### Model 2 (Port 8081)

```bash
# Health check
curl http://localhost:8081/

# Prediction (file)
curl -X POST -F "file=@image.jpg" http://localhost:8081/predict

# Prediction (URL)
curl -X POST -F "image_url=https://example.com/image.jpg" http://localhost:8081/predict
```

---

## Response Format

Both models return the same JSON structure:

```json
{
  "prediction": "FAKE",
  "confidence": 0.9613,
  "heatmap": "iVBORw0KGgoAAAANSUhEUg...",
  "model": "Model 1 (ViT Hybrid)",
  "interpretation": "Model is 96.1% confident this is FAKE. Red areas = important for decision, Blue areas = less important"
}
```

### Response Fields

| Field            | Type   | Description                      |
| ---------------- | ------ | -------------------------------- |
| `prediction`     | string | "REAL" or "FAKE"                 |
| `confidence`     | float  | Probability (0.0 - 1.0)          |
| `heatmap`        | string | Base64-encoded attention heatmap |
| `model`          | string | Which model was used             |
| `interpretation` | string | Human-readable explanation       |

---

## Heatmap Interpretation

The heatmap shows which image regions influenced the model's decision:

- 🔴 **Red/Hot areas** = Regions that increased "FAKE" prediction
- 🔵 **Blue/Cold areas** = Regions with minimal influence

**What to look for:**

**Real Images:**

- More uniform distribution
- Mostly blue areas
- Natural lighting consistency

**Deepfakes:**

- Red hotspots at artifacts
- Concentrated attention on:
  - Face boundaries (blending artifacts)
  - Eyes (unnatural reflections)
  - Jaw/mouth (temporal inconsistencies)

---

## Testing

### Test with Sample Images

```bash
# Use the web interface
1. Open http://localhost:8000/index.html
2. Select Model 1 or Model 2
3. Upload an image or paste URL
4. Compare results between models
```

### Test Images Location

```
backend/test_images/
```

---

## Troubleshooting

### Model 1 not loading

```
❌ Error: Could not load checkpoint
✅ Solution: Ensure models/vit_baseline_best.pt exists
```

### Model 2 slow on CPU

```
⏳ Time: 10+ seconds per image
✅ Solution: Use GPU (CUDA available? Check torch.cuda.is_available())
```

### CORS errors in browser

```
❌ Error: No 'Access-Control-Allow-Origin' header
✅ Solution: Both models have CORS middleware enabled, should work
```

### Port already in use

```
❌ Error: Address already in use (port 8000)
✅ Solution:
   kill -9 $(lsof -t -i :8000)
   kill -9 $(lsof -t -i :8081)
```

---

## Performance Comparison

| Aspect        | Model 1            | Model 2              |
| ------------- | ------------------ | -------------------- |
| **Speed**     | ⚡ Fast (0.3-0.5s) | 🐢 Slow (2-5s)       |
| **Accuracy**  | 🎯 Good            | 🎯🎯 Better          |
| **Resource**  | 💚 Low (MPS/CPU)   | 🔴 High (GPU needed) |
| **Artifacts** | Pixel differences  | Diffusion analysis   |
| **Best for**  | Real-time          | Offline analysis     |

---

## Next Steps

1. ✅ Both models running
2. ✅ Test with sample images
3. ✅ Compare predictions between models
4. 📊 Analyze heatmaps to understand decisions
5. 🚀 Deploy to production (Docker/cloud)

---

## Notes

- Keep both terminals running for full functionality
- Model 2 downloads diffusion weights on first run (~3GB)
- Heatmaps are gradient-based saliency maps
- DNF extraction requires diffusion models (compute-intensive)

**Questions?** Check logs in each terminal for debugging.
