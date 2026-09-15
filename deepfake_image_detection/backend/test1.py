import torch
import torch.nn as nn
import timm
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

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

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_test_images(test_root):
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
            labels.extend([1] * len(files))
            print(f"  FAKE ← {folder.relative_to(test_root)}: {len(files)} images")
        else:
            print(f"  ⚠️  Not found: {folder.relative_to(test_root)}")

    for folder in real_folders:
        if folder.exists():
            files = [f for f in folder.rglob("*") if f.suffix.lower() in exts]
            images.extend(files)
            labels.extend([0] * len(files))
            print(f"  REAL ← {folder.relative_to(test_root)}: {len(files)} images")
        else:
            print(f"  ⚠️  Not found: {folder.relative_to(test_root)}")

    print(f"\n  Total → {len(images)} images "
          f"(Real: {labels.count(0)}, Fake: {labels.count(1)})")

    if len(images) == 0:
        raise FileNotFoundError(f"No images found under {test_root}")
    return images, labels


def evaluate_checkpoint(ckpt_path, test_root, threshold=0.5, num_workers=8):
    print(f"\n{'='*55}")
    print(f"Checkpoint: {Path(ckpt_path).name}")
    print(f"{'='*55}")

    model = HybridModel().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print("✅ Weights loaded")

    print("\nLoading test images...")
    image_paths, labels = load_test_images(test_root)
    print(f"Total: {len(image_paths)} images  "
          f"(Real: {labels.count(0)}, Fake: {labels.count(1)})")

    print(f"\n🚀 Running inference with {num_workers} workers...")
    start_time = time.time()

    all_probs = [None] * len(image_paths)
    all_labels = labels.copy()
    errors = []
    processed = 0
    processed_lock = threading.Lock()

    def process_image(args):
        nonlocal processed
        idx, img_path, label = args
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = tf(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                with torch.autocast("mps" if DEVICE == "mps" else "cpu"):
                    logit = model(tensor)
                raw_prob = torch.sigmoid(logit).float().item()

            # ✅ FIX: flip probability to correct inverted labels from training
            prob = 1.0 - raw_prob

            with processed_lock:
                processed += 1
                if processed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    remaining = (len(image_paths) - processed) / rate
                    print(f"  Processed {processed}/{len(image_paths)} "
                          f"({rate:.1f} img/s, ~{remaining:.0f}s remaining)...",
                          flush=True)
            return idx, prob, None
        except Exception as e:
            return idx, None, (str(img_path), str(e))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_image, (i, img_path, label))
            for i, (img_path, label) in enumerate(zip(image_paths, labels))
        ]
        for future in as_completed(futures):
            idx, prob, error = future.result()
            if error:
                errors.append(error)
            else:
                all_probs[idx] = prob

    elapsed = time.time() - start_time
    print(f"✅ Inference completed in {elapsed:.1f}s")

    if errors:
        print(f"⚠️  Skipped {len(errors)} images due to errors")

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs > threshold).astype(int)

    auc      = roc_auc_score(all_labels, all_probs)
    acc      = (preds == all_labels).mean()
    real_mask = all_labels == 0
    fake_mask = all_labels == 1
    real_acc  = (preds[real_mask] == 0).mean() if real_mask.sum() > 0 else float('nan')
    fake_acc  = (preds[fake_mask] == 1).mean() if fake_mask.sum() > 0 else float('nan')

    print(f"\n── Results (threshold={threshold}) ──")
    print(f"  AUC-ROC:       {auc:.4f}")
    print(f"  Accuracy:      {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Real Acc:      {real_acc:.4f}  ({real_acc*100:.1f}%)")
    print(f"  Fake Acc:      {fake_acc:.4f}  ({fake_acc*100:.1f}%)")

    print(f"\n── Confusion Matrix ──")
    cm = confusion_matrix(all_labels, preds)
    print(f"             Pred:Real  Pred:Fake")
    print(f"  True:Real    {cm[0][0]:5d}      {cm[0][1]:5d}")
    print(f"  True:Fake    {cm[1][0]:5d}      {cm[1][1]:5d}")

    print(f"\n── Classification Report ──")
    print(classification_report(all_labels, preds,
                                target_names=["Real", "Fake"], digits=4))

    print(f"── Confidence Distribution ──")
    print(f"  Fake images — avg prob: {all_probs[fake_mask].mean():.3f}  "
          f"std: {all_probs[fake_mask].std():.3f}")
    print(f"  Real images — avg prob: {all_probs[real_mask].mean():.3f}  "
          f"std: {all_probs[real_mask].std():.3f}")

    return auc, acc


TEST_ROOT   = "/Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend/test_images"
CKPT_PHASE1 = "./models/model1/vit_baseline_best.pt"
CKPT_PHASE2 = "./models/model1/vit_finetuned.pt"

print(f"\n{'='*55}")
print(f"🚀 PARALLEL EVALUATION - {os.cpu_count()} CPU cores available")
print(f"{'='*55}")

start_total = time.time()
results = {}

auc1, acc1 = evaluate_checkpoint(CKPT_PHASE1, TEST_ROOT, num_workers=8)
results["Phase1 (frozen ViT)"] = (auc1, acc1)

auc2, acc2 = evaluate_checkpoint(CKPT_PHASE2, TEST_ROOT, num_workers=8)
results["Phase2 (unfrozen ViT)"] = (auc2, acc2)

total_time = time.time() - start_total

print(f"\n{'='*55}")
print(f"{'FINAL COMPARISON':^55}")
print(f"{'='*55}")
print(f"{'Checkpoint':<25}  {'AUC':>8}  {'Accuracy':>9}")
print(f"{'─'*45}")
for name, (auc, acc) in results.items():
    winner = " ← BEST" if auc == max(r[0] for r in results.values()) else ""
    print(f"{name:<25}  {auc:>8.4f}  {acc*100:>8.1f}%{winner}")
print(f"\n{'─'*55}")
print(f"Total time: {total_time:.1f}s")
print(f"{'='*55}")