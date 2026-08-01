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
| API Gateway | `app/main.py` | FastAPI app exposing all modules with API-key authentication |

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

## Ethics Note — Face Recognition

Facial recognition in a retail context raises real privacy and fairness concerns that should be addressed before any production deployment:

- **Consent**: customers should be explicitly informed and opt in before their face is enrolled or matched against a database. This prototype uses only sample images from consenting participants.
- **Data privacy**: face encodings are biometric data. In a real deployment these must be encrypted at rest, access-controlled, and covered by a clear retention/deletion policy.
- **Bias**: face recognition models (including Haar cascades and LBPH) are known to perform less reliably across different skin tones, lighting conditions, and facial features if not trained/tested on diverse data. This prototype's small sample dataset is not sufficient to evaluate or guarantee fairness — a production system would need a much larger, demographically diverse dataset and fairness auditing.
- **Purpose limitation**: recognition data should only be used for the stated purpose (e.g., loyalty recognition), not repurposed for surveillance or shared with third parties without consent.

## Scope & Limitations (compressed 4-day build)

This project was intentionally scoped down from the full 9-day plan to fit a 4-day timeline. Cut for time:
- Product image classification module (`/classify-product` endpoint, MobileNetV2 classifier) — not implemented
- DistilBERT sentiment upgrade — used TF-IDF + Logistic Regression baseline only
- CI/CD pipeline, automated tests, WebSocket live video — not implemented

These are documented here rather than left silently missing, and would be natural next steps for a full production build.

## Tech Stack

FastAPI, OpenCV (opencv-contrib-python), scikit-learn, pandas, Pydantic, Uvicorn, Docker
