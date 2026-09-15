# quick_test.py — test on your own training images
import torch, requests
from pathlib import Path

# Pick 5 fake + 5 real from YOUR training dataset
test_cases = [
    # (image_path, expected_label)
    ("test_images/Human Faces/AI-Generated Images/000001.jpg", "FAKE"),
    ("test_images/Human Faces/Real Images/000001.jpg", "REAL"),
]

import requests

for path, expected in test_cases:
    with open(path, "rb") as f:
        r = requests.post("http://localhost:8081/predict", files={"file": f})
    result = r.json()
    status = "✅" if result["prediction"] == expected else "❌"
    print(f"{status} Expected: {expected} | Got: {result['prediction']} | "
          f"prob={result['probability']:.3f} | file={Path(path).name}")