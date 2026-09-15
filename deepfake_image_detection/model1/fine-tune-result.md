(base) anupdangi~$python fine_tune.py

── Scanning folders ──
FAKE ← Human Faces/AI-Generated Images: 4630
FAKE ← ai_generatedimages/jpeg: 308
FAKE ← ai_generatedimages/png: 60
FAKE ← ai_images/test/fake: 389
FAKE ← ai_images/train/fake: 153
REAL ← Human Faces/Real Images: 5000
REAL ← ai_images/test/real: 110
REAL ← ai_images/train/real: 326

Total: 10976 | Real: 5436 | Fake: 5540

Train: 8780 | Val: 2196

✅ Loaded checkpoint: /Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend/models/model1/vit_baseline_best.pt
Trainable params: 21,956,738

── Fine-tuning on new data (8 epochs) ──
Epoch Loss Val AUC Time
──────────────────────────────────────
Epoch 1 | Batch 0/275 | Loss: 1.0295
Epoch 1 | Batch 20/275 | Loss: 0.1149
Epoch 1 | Batch 40/275 | Loss: 0.0589
Epoch 1 | Batch 60/275 | Loss: 0.2353
/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/PIL/Image.py:1039: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
warnings.warn(
Epoch 1 | Batch 80/275 | Loss: 0.0237
Epoch 1 | Batch 100/275 | Loss: 0.0346
Epoch 1 | Batch 120/275 | Loss: 0.1611
Epoch 1 | Batch 140/275 | Loss: 0.0273
Epoch 1 | Batch 160/275 | Loss: 0.0447
Epoch 1 | Batch 180/275 | Loss: 0.0876
Epoch 1 | Batch 200/275 | Loss: 0.0263
Epoch 1 | Batch 220/275 | Loss: 0.1229
Epoch 1 | Batch 240/275 | Loss: 0.0490
Epoch 1 | Batch 260/275 | Loss: 0.0051
Val AUC=0.9950 Acc=0.9567
1 0.1614 0.9950 1019s ★
💾 Saved → /Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend/models/model1/vit_finetuned.pt
Epoch 2 | Batch 0/275 | Loss: 0.0853
Epoch 2 | Batch 20/275 | Loss: 0.0036
Epoch 2 | Batch 40/275 | Loss: 0.0394
Epoch 2 | Batch 60/275 | Loss: 0.0829
Epoch 2 | Batch 80/275 | Loss: 0.0269
Epoch 2 | Batch 100/275 | Loss: 0.1864
/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/PIL/Image.py:1039: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
warnings.warn(
Epoch 2 | Batch 120/275 | Loss: 0.0093
Epoch 2 | Batch 140/275 | Loss: 0.0665
Epoch 2 | Batch 160/275 | Loss: 0.0441
Epoch 2 | Batch 180/275 | Loss: 0.0043
Epoch 2 | Batch 200/275 | Loss: 0.1902
Epoch 2 | Batch 220/275 | Loss: 0.0645
Epoch 2 | Batch 240/275 | Loss: 0.0432
Epoch 2 | Batch 260/275 | Loss: 0.1170
Val AUC=0.9968 Acc=0.9672
2 0.0552 0.9968 570s ★
💾 Saved → /Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend/models/model1/vit_finetuned.pt
Epoch 3 | Batch 0/275 | Loss: 0.0195
Epoch 3 | Batch 20/275 | Loss: 0.0079
Epoch 3 | Batch 40/275 | Loss: 0.0166
/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/PIL/Image.py:1039: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
warnings.warn(
Epoch 3 | Batch 60/275 | Loss: 0.0467
Epoch 3 | Batch 80/275 | Loss: 0.0162
Epoch 3 | Batch 100/275 | Loss: 0.0048
Epoch 3 | Batch 120/275 | Loss: 0.0137
Epoch 3 | Batch 140/275 | Loss: 0.0095
Epoch 3 | Batch 160/275 | Loss: 0.0430
Epoch 3 | Batch 180/275 | Loss: 0.0669
Epoch 3 | Batch 200/275 | Loss: 0.0102
Epoch 3 | Batch 220/275 | Loss: 0.0122
Epoch 3 | Batch 240/275 | Loss: 0.0462
Epoch 3 | Batch 260/275 | Loss: 0.0284
Val AUC=0.9973 Acc=0.9690
3 0.0335 0.9973 617s ★
💾 Saved → /Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend/models/model1/vit_finetuned.pt
Epoch 4 | Batch 0/275 | Loss: 0.0718
Epoch 4 | Batch 20/275 | Loss: 0.0020
Epoch 4 | Batch 40/275 | Loss: 0.0192
Epoch 4 | Batch 60/275 | Loss: 0.0259
Epoch 4 | Batch 80/275 | Loss: 0.0176
Epoch 4 | Batch 100/275 | Loss: 0.0322
Epoch 4 | Batch 120/275 | Loss: 0.0031
Epoch 4 | Batch 140/275 | Loss: 0.0509
Epoch 4 | Batch 160/275 | Loss: 0.0347
Epoch 4 | Batch 180/275 | Loss: 0.0010
Epoch 4 | Batch 200/275 | Loss: 0.0013
Epoch 4 | Batch 220/275 | Loss: 0.0376
/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/PIL/Image.py:1039: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
warnings.warn(
Epoch 4 | Batch 240/275 | Loss: 0.0151
Epoch 4 | Batch 260/275 | Loss: 0.0250
Val AUC=0.9976 Acc=0.9713
4 0.0217 0.9976 595s ★
💾 Saved → /Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend/models/model1/vit_finetuned.pt
Epoch 5 | Batch 0/275 | Loss: 0.0109
Epoch 5 | Batch 20/275 | Loss: 0.0038
Epoch 5 | Batch 40/275 | Loss: 0.0070
Epoch 5 | Batch 60/275 | Loss: 0.0043
Epoch 5 | Batch 80/275 | Loss: 0.0142
Epoch 5 | Batch 100/275 | Loss: 0.0374
Epoch 5 | Batch 120/275 | Loss: 0.0034
/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/PIL/Image.py:1039: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
warnings.warn(
Epoch 5 | Batch 140/275 | Loss: 0.0333
Epoch 5 | Batch 160/275 | Loss: 0.0017
Epoch 5 | Batch 180/275 | Loss: 0.0099
Epoch 5 | Batch 200/275 | Loss: 0.0058
Epoch 5 | Batch 220/275 | Loss: 0.0105
Epoch 5 | Batch 240/275 | Loss: 0.0054
Epoch 5 | Batch 260/275 | Loss: 0.0117
Val AUC=0.9977 Acc=0.9736
5 0.0138 0.9977 621s ★
💾 Saved → /Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend/models/model1/vit_finetuned.pt
Epoch 6 | Batch 0/275 | Loss: 0.0017
Epoch 6 | Batch 20/275 | Loss: 0.0008
Epoch 6 | Batch 40/275 | Loss: 0.0026
Epoch 6 | Batch 60/275 | Loss: 0.0004
Epoch 6 | Batch 80/275 | Loss: 0.0129
Epoch 6 | Batch 100/275 | Loss: 0.0249
Epoch 6 | Batch 120/275 | Loss: 0.0094
Epoch 6 | Batch 140/275 | Loss: 0.0078
Epoch 6 | Batch 160/275 | Loss: 0.0348
Epoch 6 | Batch 180/275 | Loss: 0.0058
/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/PIL/Image.py:1039: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
warnings.warn(
Epoch 6 | Batch 200/275 | Loss: 0.0083
Epoch 6 | Batch 220/275 | Loss: 0.0037
Epoch 6 | Batch 240/275 | Loss: 0.0097
Epoch 6 | Batch 260/275 | Loss: 0.0173
Val AUC=0.9978 Acc=0.9745
6 0.0107 0.9978 602s ★
💾 Saved → /Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend/models/model1/vit_finetuned.pt
Epoch 7 | Batch 0/275 | Loss: 0.0343
Epoch 7 | Batch 20/275 | Loss: 0.0007
/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/PIL/Image.py:1039: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
warnings.warn(
Epoch 7 | Batch 40/275 | Loss: 0.0021
Epoch 7 | Batch 60/275 | Loss: 0.0279
Epoch 7 | Batch 80/275 | Loss: 0.0202
Epoch 7 | Batch 100/275 | Loss: 0.0106
Epoch 7 | Batch 120/275 | Loss: 0.0142
Epoch 7 | Batch 140/275 | Loss: 0.0063
Epoch 7 | Batch 160/275 | Loss: 0.0027
Epoch 7 | Batch 180/275 | Loss: 0.0015
Epoch 7 | Batch 200/275 | Loss: 0.0002
Epoch 7 | Batch 220/275 | Loss: 0.0031
Epoch 7 | Batch 240/275 | Loss: 0.0028
Epoch 7 | Batch 260/275 | Loss: 0.0044
Val AUC=0.9979 Acc=0.9750
7 0.0087 0.9979 649s ★
💾 Saved → /Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend/models/model1/vit_finetuned.pt
Epoch 8 | Batch 0/275 | Loss: 0.0005
Epoch 8 | Batch 20/275 | Loss: 0.0268
Epoch 8 | Batch 40/275 | Loss: 0.0126
Epoch 8 | Batch 60/275 | Loss: 0.0189
Epoch 8 | Batch 80/275 | Loss: 0.0177
Epoch 8 | Batch 100/275 | Loss: 0.0179
Epoch 8 | Batch 120/275 | Loss: 0.0100
/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/PIL/Image.py:1039: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
warnings.warn(
Epoch 8 | Batch 140/275 | Loss: 0.0059
Epoch 8 | Batch 160/275 | Loss: 0.0023
Epoch 8 | Batch 180/275 | Loss: 0.0080
Epoch 8 | Batch 200/275 | Loss: 0.0006
Epoch 8 | Batch 220/275 | Loss: 0.0079
Epoch 8 | Batch 240/275 | Loss: 0.0039
Epoch 8 | Batch 260/275 | Loss: 0.0071
Val AUC=0.9979 Acc=0.9750
8 0.0073 0.9979 723s

✅ Fine-tuning done. Best Val AUC: 0.9979
Weights → /Users/anupdangi/Desktop/AnupAI/projects/Internship_project/deepfake_image_detection/backend/models/model1/vit_finetuned.pt
