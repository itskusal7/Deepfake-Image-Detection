import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from pathlib import Path
from PIL import Image
import numpy as np
import time

# ── Config ───────────────────────────────────────────────────
DEVICE    = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_IN  = "/Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend/models/model1/vit_baseline_best.pt"
CKPT_OUT = "/Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend/models/model1/vit_finetuned.pt"
TEST_ROOT = "/Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend/test_images"

BATCH_SIZE = 32
EPOCHS     = 8
LR         = 5e-6    # very low — we're fine-tuning, not training from scratch

# ── Model (same as before) ───────────────────────────────────
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

# ── Custom Dataset ────────────────────────────────────────────
class ImageListDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            return self.transform(img), self.labels[idx]
        except:
            # return a black image if corrupt
            return torch.zeros(3, 224, 224), self.labels[idx]

# ── Load all images from test_images folder ───────────────────
def load_all_images(test_root):
    test_root = Path(test_root)
    images, labels = [], []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    fake_folders = [
        test_root / "Human Faces" / "AI-Generated Images",
        test_root / "ai_generatedimages" / "jpeg",
        test_root / "ai_generatedimages" / "png",
        test_root / "ai_images" / "test"  / "fake",
        test_root / "ai_images" / "train" / "fake",
    ]
    real_folders = [
        test_root / "Human Faces" / "Real Images",
        test_root / "ai_images" / "test"  / "real",
        test_root / "ai_images" / "train" / "real",
    ]

    print("\n── Scanning folders ──")
    for folder in fake_folders:
        if folder.exists():
            files = [f for f in folder.rglob("*") if f.suffix.lower() in exts]
            images.extend(files)
            # ✅ IMPORTANT: inverted labels to match how model was trained
            # model trained with fake=0,real=1 → flip at inference with 1-prob
            # but for fine-tuning we correct it: fake=1, real=0 properly
            labels.extend([1] * len(files))
            print(f"  FAKE ← {folder.relative_to(test_root)}: {len(files)}")

    for folder in real_folders:
        if folder.exists():
            files = [f for f in folder.rglob("*") if f.suffix.lower() in exts]
            images.extend(files)
            labels.extend([0] * len(files))
            print(f"  REAL ← {folder.relative_to(test_root)}: {len(files)}")

    print(f"\n  Total: {len(images)} | Real: {labels.count(0)} | Fake: {labels.count(1)}")
    return images, labels

# ── Train/Val split (80/20) ───────────────────────────────────
from sklearn.model_selection import train_test_split

all_images, all_labels = load_all_images(TEST_ROOT)

train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
    all_images, all_labels,
    test_size=0.2,
    random_state=42,
    stratify=all_labels    # keeps fake/real ratio balanced
)
print(f"\nTrain: {len(train_imgs)} | Val: {len(val_imgs)}")

# ── Transforms ────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_ds = ImageListDataset(train_imgs, train_lbls, train_tf)
val_ds   = ImageListDataset(val_imgs,   val_lbls,   val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)

# ── Load existing checkpoint ──────────────────────────────────
model = HybridModel().to(DEVICE)
state = torch.load(CKPT_IN, map_location=DEVICE)
model.load_state_dict(state)
print(f"\n✅ Loaded checkpoint: {CKPT_IN}")

# ── Unfreeze everything for fine-tuning ───────────────────────
for p in model.parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Trainable params: {trainable:,}")

# ── Optimizer — very low LR to preserve learned weights ───────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-7
)
criterion = nn.BCEWithLogitsLoss()

# ── Evaluate ──────────────────────────────────────────────────
def evaluate(loader, name):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            with torch.autocast("mps"):
                logits = model(imgs)
            # ✅ flip prob — model was trained with inverted labels
            probs = 1.0 - torch.sigmoid(logits).float().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    auc = roc_auc_score(all_labels, all_probs)
    acc = ((np.array(all_probs) > 0.5).astype(int) == np.array(all_labels)).mean()
    print(f"  {name:15s}  AUC={auc:.4f}  Acc={acc:.4f}")
    return auc

# ── Fine-tuning loop ──────────────────────────────────────────
print(f"\n── Fine-tuning on new data ({EPOCHS} epochs) ──")
print(f"{'Epoch':>5}  {'Loss':>8}  {'Val AUC':>8}  {'Time':>6}")
print("─" * 38)

best_auc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs   = imgs.to(DEVICE)
        # ✅ flip labels to match model's internal convention
        labels = (1.0 - labels.float()).to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("mps"):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} "
                  f"| Loss: {loss.item():.4f}", flush=True)

    scheduler.step()
    va_auc = evaluate(val_loader, "Val")
    elapsed = time.time() - t0
    marker  = " ★" if va_auc > best_auc else ""

    print(f"{epoch:>5}  {total_loss/len(train_loader):>8.4f}  "
          f"{va_auc:>8.4f}  {elapsed:>5.0f}s{marker}")

    if va_auc > best_auc:
        best_auc = va_auc
        torch.save(model.state_dict(), CKPT_OUT)
        print(f"  💾 Saved → {CKPT_OUT}")

print(f"\n✅ Fine-tuning done. Best Val AUC: {best_auc:.4f}")
print(f"   Weights → {CKPT_OUT}")