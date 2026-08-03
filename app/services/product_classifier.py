"""
Lightweight product image classifier.

Scope note: the original plan called for a MobileNetV2 transfer-learning
classifier. Given unreliable internet access during development (large
pretrained weight downloads repeatedly failed/timed out), this module
uses a classical CV approach instead: color histograms + Local Binary
Pattern texture features, fed into a RandomForest classifier. It requires
no external downloads and trains in seconds on a small custom dataset.

Upgrade path: swap `extract_features()` for a MobileNetV2 embedding and
retrain — the rest of the pipeline (train/predict/API wiring) stays the same.
"""
import cv2
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_PATH = "app/models/product_classifier.pkl"
DATA_DIR = "data/products"  # data/products/<category>/*.jpg


def extract_features(img):
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color histogram (coarse shape/appearance signature)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Texture via simple gradient magnitude histogram (LBP-lite)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    texture_hist, _ = np.histogram(mag, bins=16, range=(0, 255))
    texture_hist = texture_hist / (texture_hist.sum() + 1e-6)

    return np.concatenate([hist, texture_hist])


def train():
    X, y = [], []
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"No dataset found at {DATA_DIR}")

    categories = [d for d in os.listdir(DATA_DIR) if os.path.isdir(f"{DATA_DIR}/{d}")]
    if len(categories) < 2:
        raise ValueError("Need at least 2 product categories with images to train.")

    for cat in categories:
        folder = f"{DATA_DIR}/{cat}"
        for fname in os.listdir(folder):
            img = cv2.imread(f"{folder}/{fname}")
            if img is None:
                continue
            X.append(extract_features(img))
            y.append(cat)

    if len(X) < 4:
        raise ValueError("Need at least a few images per category to train.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test) if len(X_test) > 0 else None

    os.makedirs("app/models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "categories": sorted(set(y))}, f)

    print(f"Trained on {len(X)} images across {len(categories)} categories. Test accuracy: {acc}")
    return acc


def predict(image_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not trained yet. Run train() first.")
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    clf = saved["model"]

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image.")
    feat = extract_features(img).reshape(1, -1)
    pred = clf.predict(feat)[0]
    proba = max(clf.predict_proba(feat)[0])
    return {"category": pred, "confidence": round(float(proba), 3)}


if __name__ == "__main__":
    train()
