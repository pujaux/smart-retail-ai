import cv2
import os
import pickle
import numpy as np
from datetime import datetime

DB_DIR = "app/models/faces"
LABELS_PATH = "app/models/labels.pkl"
MODEL_PATH = "app/models/lbph_model.yml"

CASCADE_PATH = "app/models/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

def _detect_face_roi(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return cv2.resize(gray[y:y+h, x:x+w], (200, 200))

def enroll_customer(name, image_path):
    os.makedirs(DB_DIR, exist_ok=True)
    roi = _detect_face_roi(image_path)
    if roi is None:
        return False
    cv2.imwrite(f"{DB_DIR}/{name}.jpg", roi)
    _retrain()
    return True

def _retrain():
    faces, labels, names = [], [], []
    if not os.path.exists(DB_DIR):
        return
    for i, fname in enumerate(os.listdir(DB_DIR)):
        img = cv2.imread(f"{DB_DIR}/{fname}", cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(i)
        names.append(fname.replace(".jpg", ""))
    if not faces:
        return
    recognizer.train(faces, np.array(labels))
    os.makedirs("app/models", exist_ok=True)
    recognizer.save(MODEL_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(names, f)

def recognize(image_path, threshold=80):
    if not os.path.exists(MODEL_PATH):
        return {"match": False, "name": None}
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH, "rb") as f:
        names = pickle.load(f)
    roi = _detect_face_roi(image_path)
    if roi is None:
        return {"match": False, "name": None}
    label, confidence = recognizer.predict(roi)  # lower confidence = better match
    if confidence < threshold:
        return {
            "match": True,
            "name": names[label],
            "confidence": round(100 - confidence, 1),
            "timestamp": datetime.now().isoformat()
        }
    return {"match": False, "name": None}

if __name__ == "__main__":
    enroll_customer("alice", "data/alice.jpg")
    print(recognize("data/test.jpg"))
