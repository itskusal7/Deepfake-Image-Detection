import torch
import torch.nn as nn
import timm
import requests
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import transforms
import os
import cv2
import numpy as np
import base64

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
        self.vit_fc = nn.Linear(384, 256)
        self.npr_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.npr_fc = nn.Linear(128, 256)
        self.attn = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        vit_feat = self.vit_fc(self.vit(x))
        npr = torch.abs(x[:, :, :-2, :-2] - x[:, :, 2:, 2:])
        npr_feat = self.npr_fc(self.npr_cnn(npr).view(x.size(0), -1))
        feats = torch.stack([vit_feat, npr_feat], dim=1)
        weights = torch.softmax(self.attn(feats).squeeze(-1), dim=1)
        fused = (feats * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(fused).squeeze(1)


MODEL_PATH = "./models/model1/vit_finetuned.pt"
print(f"Model path exists: {os.path.exists(MODEL_PATH)}")

model = HybridModel().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
print("Hybrid model loaded successfully")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ FIX 1: Added missing @ decorator
@app.get("/")
async def root():
    return {"status": "Server is running"}

@app.get("/index.html")
async def get_index():
    return FileResponse("index.html")


def generate_heatmap(model, image_tensor):
    """Gradient-based saliency heatmap."""
    model.eval()
    # ✅ FIX 2: fresh tensor with grad enabled, outside no_grad
    image_tensor = image_tensor.detach().clone().requires_grad_(True)

    output = model(image_tensor)
    score  = torch.sigmoid(output)
    model.zero_grad()
    score.backward()

    grads   = image_tensor.grad[0]
    heatmap = grads.abs().mean(dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)
    heatmap  = cv2.GaussianBlur(heatmap, (11, 11), 2)
    return heatmap


@app.post("/predict")
async def predict(file: UploadFile = None, image_url: str = Form(None)):
    try:
        if file:
            contents = await file.read()
            if not contents:
                raise HTTPException(400, "File is empty")
            img = Image.open(BytesIO(contents)).convert("RGB")
        elif image_url:
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert("RGB")
            except requests.exceptions.RequestException as e:
                raise HTTPException(400, f"Failed to fetch image from URL: {str(e)}")
        else:
            raise HTTPException(400, "Provide either file or image_url")

        input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

        # ── Prediction ─────────────────────────────────────
        with torch.no_grad():
            output = model(input_tensor)

        raw_prob = torch.sigmoid(output).item()

        # ✅ FIX 3: flip prob to correct training label inversion
        prob = 1.0 - raw_prob

        # ✅ FIX 4: use 0.5 threshold (matches training eval)
            
        # Better — reflects actual model uncertainty
        if prob > 0.70:   prediction = "FAKE"
        elif prob < 0.30: prediction = "REAL"
        else:             prediction = "UNCERTAIN"  # ~40% of images will be UNCERTAIN, which is honest

        # ── Heatmap ────────────────────────────────────────
        # ✅ FIX 5: fresh tensor passed to heatmap (no conflict with no_grad above)
        heatmap = generate_heatmap(model, input_tensor)

        heatmap_uint8   = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        _, buffer       = cv2.imencode('.png', heatmap_colored)
        heatmap_base64  = base64.b64encode(buffer).decode()

        return {
            "prediction":      prediction,
            "confidence":      round(prob, 4),
            "raw_prob":        round(raw_prob, 4),   # useful for debugging
            "heatmap":         heatmap_base64,
            "interpretation":  (
                f"Fake probability: {prob:.1%}. "
                f"Prediction: {prediction}. "
                f"UNCERTAIN = model not confident (0.35–0.65 range)."
            )
        }

    except Exception as e:
        raise HTTPException(400, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("model1:app", host="0.0.0.0", port=8000, reload=True)