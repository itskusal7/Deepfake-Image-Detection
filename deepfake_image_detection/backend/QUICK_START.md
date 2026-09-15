# ⚡ Quick Start Guide

## 3-Step Setup

### Step 1: Make startup script executable

```bash
chmod +x /Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend/startup.sh
```

### Step 2: Run both models

```bash
cd /Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend
./startup.sh
```

### Step 3: Open browser

```
http://localhost:8000/index.html
```

---

## Manual Setup (if script fails)

### Terminal 1: Model 1 (ViT Hybrid)

```bash
cd backend
python main.py
```

### Terminal 2: Model 2 (DNF-Based)

```bash
cd backend
python model2.py
```

### Terminal 3: Open browser

```
http://localhost:8000/index.html
```

---

## What You'll See

### Model Selection Panel

```
📋 Choose Detection Model

○ Model 1: ViT Hybrid
  ViT + NPR with attention fusion. Fast & accurate.

○ Model 2: DNF-Based
  Diffusion-based features + NPR. More accurate.

✅ Model 1 on port 8000 | ⏳ Model 2 on port 8081
```

### Upload & Analyze

1. Select a model
2. Upload image or paste URL
3. Click "🚀 Analyze Image"
4. View results + heatmap

---

## Key Differences

| Feature      | Model 1     | Model 2       |
| ------------ | ----------- | ------------- |
| **Port**     | 8000        | 8081          |
| **Speed**    | ⚡ 0.3-0.5s | 🐢 2-5s       |
| **Accuracy** | 🎯 Good     | 🎯🎯 Better   |
| **GPU**      | Optional    | Recommended   |
| **Use Case** | Real-time   | High accuracy |

---

## Heatmap Guide

- 🔴 **Red** = Important artifacts detected
- 🔵 **Blue** = Less important regions
- Distribution = How spread out the decision is

**Real images:** Mostly uniform blue
**Deepfakes:** Red hotspots at artifacts

---

## Troubleshooting

### Issue: Port 8000 already in use

```bash
kill -9 $(lsof -t -i :8000)
```

### Issue: Model 2 is slow

- Ensure GPU is available
- First run downloads ~3GB diffusion models
- Next runs will be faster (cached)

### Issue: Checkpoint not found

- Download `vit_baseline_best.pt` → `models/`
- Download `detector_final.pt` → `models/`

### Issue: ModuleNotFoundError

```bash
pip install timm diffusers
```

---

## Deployment Notes

✅ Both models are CORS-enabled
✅ Automatic model status checking (every 5 sec)
✅ Base64 heatmap encoding (no file uploads needed)
✅ Error recovery with helpful messages

---

## Architecture Diagram

```
User Browser (http://localhost:8000/index.html)
       ↓
   Model Selection (radio button)
       ↓ sends request to...
   ╔═══════════════════════════════════════╗
   ║  Model 1 (Port 8000)  ←→ GPU/MPS      ║
   ║  ViT + NPR + Attention                ║
   ║  Fast processing                      ║
   ╚═══════════════════════════════════════╝
       OR
   ╔═══════════════════════════════════════╗
   ║  Model 2 (Port 8081)  ←→ GPU/Diffusion║
   ║  DNF + NPR + Fusion                   ║
   ║  Slower but more accurate             ║
   ╚═══════════════════════════════════════╝
       ↓
   JSON Response with
   - prediction (REAL/FAKE)
   - confidence (0-1)
   - heatmap (base64 image)
   - model info
       ↓
   Browser displays results
```

---

## Next: Verify Setup

After starting both models, check:

```bash
# Terminal check
curl http://localhost:8000/
curl http://localhost:8081/

# Should return:
# {"status":"ok","device":"mps",...}
```

---

**Ready to detect deepfakes!** 🎯
