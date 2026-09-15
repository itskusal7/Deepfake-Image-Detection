# main.py — DNF Deepfake Detector API
# Aligned exactly to training code

import torch
import torch.nn as nn
import torchvision.models as models
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import numpy as np
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, LCMScheduler
import base64
import cv2
import requests
import traceback
import ssl
import urllib3

# ✅ FIX: Suppress SSL warnings for URL fetching
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ═══════════════════════════════════════════
# 1. MODEL — exact copy from training
# ═══════════════════════════════════════════
FEAT_DIM = 128

class DNFBranch(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.resnet34(weights="IMAGENET1K_V1")          # no pretrained weights needed at inference
        bb.conv1 = nn.Conv2d(2, 64, 7, 2, 3, bias=False)
        self.backbone = nn.Sequential(*list(bb.children())[:-1])
        self.fc = nn.Linear(512, FEAT_DIM)

    def forward(self, x):
        return self.fc(self.backbone(x).flatten(1))

class NPRBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, FEAT_DIM)

    def forward(self, x):
        dh = torch.abs(x - torch.roll(x, 1, dims=-1))
        dv = torch.abs(x - torch.roll(x, 1, dims=-2))
        x  = (dh + dv) / 2
        return self.fc(self.net(x).flatten(1))

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(FEAT_DIM, 1)

    def forward(self, f1, f2):
        feats = torch.stack([f1, f2], dim=1)          # [B, 2, D]
        w     = torch.softmax(self.attn(feats).squeeze(-1), dim=1)
        return (w.unsqueeze(-1) * feats).sum(1)

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnf    = DNFBranch()
        self.npr    = NPRBranch()
        self.fusion = Fusion()
        self.head   = nn.Linear(FEAT_DIM, 1)          # ✅ simple Linear — matches training

    def forward(self, img, dnf):
        f1 = self.dnf(dnf)
        f2 = self.npr(img)
        f  = self.fusion(f1, f2)
        return self.head(f).squeeze(1)

# ═══════════════════════════════════════════
# 2. CONFIG
# ═══════════════════════════════════════════
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE      = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_PATH = "./models/model2/detector_final.pt"               # ← your best checkpoint
LCM_MODEL  = "SimianLuo/LCM_Dreamshaper_v7"

print(f"Device : {DEVICE}")
print(f"DType  : {DTYPE}")

# ═══════════════════════════════════════════
# 3. LOAD DETECTOR
#    Training saved: {"epoch":…, "model": state_dict, "optimizer":…, "val_auc":…}
#    So we MUST unpack ["model"]
# ═══════════════════════════════════════════
print("\nLoading detector checkpoint...")
model = Detector().to(DEVICE)

raw = torch.load(MODEL_PATH, map_location=DEVICE)

# Detect checkpoint format automatically
if isinstance(raw, dict) and "model" in raw:
    state_dict = raw["model"]
    print(f"  Format   : full checkpoint dict")
    print(f"  Epoch    : {raw.get('epoch', '?')}")
    print(f"  Val AUC  : {raw.get('val_auc', '?'):.4f}")
else:
    state_dict = raw
    print(f"  Format   : raw state dict")

try:
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        print(f"  Missing   : {missing}")
        print(f"  Unexpected: {unexpected}")
    else:
        print(f"  All weights loaded - model is ready")
except Exception as e:
    print(f"  Error loading weights: {e}")
    print(f"  Attempting non-strict load...")
    model.load_state_dict(state_dict, strict=False)
    print(f"  Model loaded (non-strict mode)")

model.eval()
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ═══════════════════════════════════════════
# 4. LOAD DIFFUSION MODELS
# ═══════════════════════════════════════════
print("\nLoading diffusion models...")
vae   = AutoencoderKL.from_pretrained(LCM_MODEL, subfolder="vae",  torch_dtype=DTYPE).to(DEVICE)
unet  = UNet2DConditionModel.from_pretrained(LCM_MODEL, subfolder="unet", torch_dtype=DTYPE).to(DEVICE)
sched = LCMScheduler.from_pretrained(LCM_MODEL, subfolder="scheduler")
sched.set_timesteps(4)
print("Diffusion models ready\n")

# ═══════════════════════════════════════════
# 5. TRANSFORMS
#    img_tf  → matches val_tf from training (no augmentation at inference)
#    dnf_tf  → matches to_tensor used in extract_dnf during training
# ═══════════════════════════════════════════
img_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),  # ✅ ImageNet norm — matches val_tf
])

dnf_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),             # ✅ matches training extract_dnf exactly
])

# ═══════════════════════════════════════════
# 6. DNF EXTRACTION
#    Must match training extract_dnf() exactly
# ═══════════════════════════════════════════
@torch.no_grad()
def extract_dnf(image: Image.Image) -> torch.Tensor:
    img    = dnf_tf(image).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    latent = vae.encode(img).latent_dist.sample()
    latent = latent * vae.config.scaling_factor

    residuals = []
    for t in sched.timesteps[:2]:                       # ✅ 2 timesteps — matches training
        noise      = torch.randn_like(latent)
        noisy      = sched.add_noise(latent, noise, t.unsqueeze(0))
        noise_pred = unet(
            noisy, t,
            encoder_hidden_states=torch.zeros(
                1, 1, unet.config.cross_attention_dim,
                device=DEVICE, dtype=DTYPE
            )
        ).sample
        residual = noise_pred - noise
        decoded  = vae.decode(residual / vae.config.scaling_factor).sample
        residuals.append(decoded)

    stack    = torch.stack(residuals)                   # [2, 1, 3, 224, 224]
    mean_map = stack.mean(0).mean(1, keepdim=True)      # [1, 1, 224, 224]
    std_map  = stack.std(0).mean(1, keepdim=True)
    dnf      = torch.cat([mean_map, std_map], dim=1)    # [1, 2, 224, 224]

    # ✅ CRITICAL — global normalization, must match training exactly
    dnf = (dnf - dnf.mean()) / (dnf.std() + 1e-6)

    return dnf.float()                                  # float32 for detector

# ═══════════════════════════════════════════
# 7. GRADIENT HEATMAP
# ═══════════════════════════════════════════
def generate_heatmap(image_tensor: torch.Tensor, dnf_tensor: torch.Tensor) -> np.ndarray:
    model.eval()
    img = image_tensor.clone().detach().requires_grad_(True)  # fresh clone with grad
    dnf = dnf_tensor.detach()

    model.zero_grad()                                   # ✅ zero BEFORE forward
    out   = model(img, dnf)
    score = torch.sigmoid(out)
    score.backward()

    grads   = img.grad[0].abs().mean(dim=0).cpu().numpy()
    p95     = np.percentile(grads, 95)
    grads   = np.clip(grads, 0, p95)
    grads   = (grads - grads.min()) / (grads.max() - grads.min() + 1e-8)
    grads   = cv2.resize(grads, (224, 224))
    grads   = cv2.GaussianBlur(grads, (5, 5), 1.0)
    grads   = (grads - grads.min()) / (grads.max() - grads.min() + 1e-8)
    return grads

# ═══════════════════════════════════════════
# 8. FASTAPI
# ═══════════════════════════════════════════
app = FastAPI(title="DNF Deepfake Detector", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {
        "status" : "ok",
        "model"  : "DNF + NPR Detector",
        "device" : DEVICE,
        "port"   : 8081
    }

@app.post("/predict")
async def predict(
    file      : UploadFile = File(None),
    image_url : str        = Form(None),
):
    try:
        # ── Load image ──────────────────────────────────────
        if file and file.filename:
            contents = await file.read()
            if not contents:
                raise HTTPException(400, "Uploaded file is empty")
            img = Image.open(BytesIO(contents)).convert("RGB")

        elif image_url:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                resp = requests.get(image_url, timeout=10, headers=headers, verify=False)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
            except requests.RequestException as e:
                raise HTTPException(400, f"Failed to fetch image: {e}")
            except Exception as e:
                raise HTTPException(400, f"Image processing error: {str(e)}")
        else:
            raise HTTPException(400, "Send either 'file' (multipart) or 'image_url' (form field)")

        # ── Inference ───────────────────────────────────────
        img_tensor = img_tf(img).unsqueeze(0).to(DEVICE)   # [1,3,224,224]
        dnf        = extract_dnf(img)                       # [1,2,224,224] float32
        
        # ✅ DEBUG: Check DNF values
        print(f"[DEBUG] DNF Shape: {dnf.shape}, DNF Mean: {dnf.mean().item():.6f}, DNF Std: {dnf.std().item():.6f}")

        with torch.no_grad():
            logit = model(img_tensor, dnf)
            prob  = torch.sigmoid(logit).item()             # 0.0 → REAL, 1.0 → FAKE
        
        # ✅ DEBUG: Print logit and probability
        print(f"[DEBUG] Logit: {logit.item():.6f}, Sigmoid(Logit): {prob:.6f}")
        
        # ✅ Check if model is in eval mode
        print(f"[DEBUG] Model training mode: {model.training}")

        prediction = "FAKE" if prob > 0.5 else "REAL"
        confidence = abs(prob - 0.5) * 2  # 0-1 scale

        # ── Heatmap (needs grad — use fresh tensor) ─────────
        img_for_heatmap = img_tf(img).unsqueeze(0).to(DEVICE)
        heatmap         = generate_heatmap(img_for_heatmap, dnf)

        hm_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        _, buf     = cv2.imencode(".png", hm_colored)
        hm_b64     = base64.b64encode(buf).decode()

        return {
            "prediction"    : prediction,
            "confidence"    : round(confidence, 4),
            "probability"   : round(prob, 4),
            "fake_score"    : round(prob, 4),       # prob=1 means FAKE
            "real_score"    : round(1 - prob, 4),
            "heatmap"       : hm_b64,
            "model"         : "DNF + NPR Detector",
            "port"          : 8081,
            "interpretation": f"{prob*100:.1f}% FAKE confidence",
        }

    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "Internal server error — check server logs")

# ✅ ADD ALIAS ENDPOINT for /detect
@app.post("/detect")
async def detect(
    file      : UploadFile = File(None),
    image_url : str        = Form(None),
):
    """Alias for /predict endpoint"""
    return await predict(file, image_url)

# ═══════════════════════════════════════════
# 9. RUN
# ═══════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("DNF Detector - Port 8081")
    print("=" * 60)
    uvicorn.run("model2:app", host="0.0.0.0", port=8081,reload=True)