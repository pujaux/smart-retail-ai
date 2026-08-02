# Smart Retail & Customer Intelligence Platform

An AI-powered platform for retail businesses that recognizes returning customers via face recognition, analyzes customer sentiment, and answers FAQs through a chatbot — all exposed through a single FastAPI backend.

## Architecture

```
Client (Postman / browser / webcam)
        |  REST calls
        v
FastAPI Gateway (app/main.py)
  /recognize-face   /enroll-customer
  /analyze-sentiment   /chatbot   /dashboard/stats
        |
  -----------------------------------------
  |                        |               |
  v                        v               v
CV Module              NLP Module      Chatbot Module
- OpenCV capture        - TF-IDF        - TF-IDF intent
- Haar cascade detect   - Logistic      matching
- LBPH face recognizer  Regression      - Cosine similarity
```

## Modules

| Module | File | Description |
|---|---|---|
| Computer Vision | `app/services/cv_utils.py` | Grayscale, resize, blur, Canny edge detection, Haar cascade face detection |
| Face Recognition | `app/services/face_recognition_module.py` | OpenCV LBPH recognizer — enroll and identify returning customers |
| Sentiment Analysis | `app/services/sentiment_module.py` | TF-IDF + Logistic Regression, classifies reviews as positive/negative/neutral |
| Chatbot | `app/services/chatbot_module.py` | TF-IDF + cosine similarity intent matching over a custom `intents.json` |
| Product Classifier | `app/services/product_classifier.py` | Color histogram + gradient-texture features → RandomForest, classifies product photos (shoes/electronics/clothing) |
| API Gateway | `app/main.py` | FastAPI app exposing all modules with API-key authentication |
| Frontend | `dashboard.py` | Streamlit dashboard with a tab per module, calling the FastAPI backend |

## Testing

Automated tests cover sentiment, chatbot, and product classification logic:
```bash
pip install pytest
python -m pytest tests/ -v
```

## Setup

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Swagger docs available at: `http://127.0.0.1:8000/docs`

All protected endpoints require header: `x-api-key: retail-secret-key`

## Docker

```bash
docker build -t smart-retail-ai .
docker run -p 8000:8000 smart-retail-ai
```

## Datasets

- **Face recognition**: self-collected sample images (consenting participants), used to demonstrate enrollment and recognition. Not a production-scale dataset.
- **Sentiment analysis**: a template-generated retail review dataset (600 rows, 3 sentiment classes, 20 product categories), simulating real customer reviews for demo purposes.
- **Chatbot**: custom `intents.json` covering common retail FAQs (greetings, order status, returns, payments, discounts).
- **Product classification**: ~244 real product photos across 3 categories (shoes, electronics, clothing), sourced from a personal collection and public product images for demo purposes. Achieves ~70% test accuracy with a lightweight classical-CV model (color + texture features + RandomForest) — no deep learning weights required, so the module has no external download dependency.

## Ethics Note — Face Recognition

Facial recognition in a retail context raises real privacy and fairness concerns that should be addressed before any production deployment:

- **Consent**: customers should be explicitly informed and opt in before their face is enrolled or matched against a database. This prototype uses only sample images from consenting participants.
- **Data privacy**: face encodings are biometric data. In a real deployment these must be encrypted at rest, access-controlled, and covered by a clear retention/deletion policy.
- **Bias**: face recognition models (including Haar cascades and LBPH) are known to perform less reliably across different skin tones, lighting conditions, and facial features if not trained/tested on diverse data. This prototype's small sample dataset is not sufficient to evaluate or guarantee fairness — a production system would need a much larger, demographically diverse dataset and fairness auditing.
- **Purpose limitation**: recognition data should only be used for the stated purpose (e.g., loyalty recognition), not repurposed for surveillance or shared with third parties without consent.

## Scope & Limitations (compressed 4-day build)

This project was intentionally scoped down from the full 9-day plan to fit a 4-day timeline. What was originally cut for time was later completed once the timeline allowed:

- **DistilBERT sentiment** — `predict_distilbert()` in `sentiment_module.py`, using `distilbert-base-uncased-finetuned-sst-2-english` via Hugging Face Transformers. Available via `/analyze-sentiment` with `"use_distilbert": true`. TF-IDF + Logistic Regression remains the default (faster, no model download needed) with DistilBERT as an opt-in upgrade.
- **Product image classifier** — built with a lightweight classical-CV approach (color + texture features + RandomForest) rather than MobileNetV2, trained on ~244 real product photos, ~70% test accuracy. No external model download required.
- **Automated test suite** (`tests/test_modules.py`) covering sentiment, chatbot, and product classification.
- **Docker** — build and run verified end-to-end; all endpoints confirmed live inside the container.
- **Full Streamlit frontend dashboard** unifying all modules with custom styling.

Remaining out of scope: CI/CD pipeline, WebSocket live video streaming — not implemented, natural next steps for a full production build.

## Tech Stack

FastAPI, OpenCV (opencv-contrib-python), scikit-learn, pandas, Pydantic, Uvicorn, Docker
