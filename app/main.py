import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import shutil, os as _os

from app.pipeline import pipeline

app = FastAPI(title="Smart Retail AI Platform", version="1.0")

API_KEY = os.environ.get("RETAIL_API_KEY", "retail-secret-key")

def check_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

def safe_filename(name):
    ext = os.path.splitext(name)[1] or ".jpg"
    return f"upload_{abs(hash(name))}{ext}"

class SentimentRequest(BaseModel):
    text: str
    use_distilbert: bool = False

class ChatRequest(BaseModel):
    message: str

# ---------- Endpoints ----------

@app.get("/")
def root():
    return {"status": "Smart Retail AI Platform is running"}

@app.post("/recognize-face")
async def recognize_face(file: UploadFile = File(...), x_api_key: str = Header(None)):
    check_key(x_api_key)
    temp_path = f"data/_{safe_filename(file.filename)}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = pipeline.recognize_face(temp_path)
    _os.remove(temp_path)
    return result

@app.post("/enroll-customer")
async def enroll(name: str, file: UploadFile = File(...), x_api_key: str = Header(None)):
    check_key(x_api_key)
    temp_path = f"data/_{safe_filename(file.filename)}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    success = pipeline.enroll_customer(name, temp_path)
    _os.remove(temp_path)
    if not success:
        raise HTTPException(status_code=400, detail="No face detected in image")
    return {"enrolled": True, "name": name}

@app.post("/analyze-sentiment")
def analyze_sentiment(req: SentimentRequest, x_api_key: str = Header(None)):
    check_key(x_api_key)
    try:
        return pipeline.analyze_sentiment(req.text, use_distilbert=req.use_distilbert)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chatbot")
def chatbot_reply(req: ChatRequest, x_api_key: str = Header(None)):
    check_key(x_api_key)
    return pipeline.chat(req.message)

@app.post("/classify-product")
async def classify_product(file: UploadFile = File(...), x_api_key: str = Header(None)):
    check_key(x_api_key)
    temp_path = f"data/_{safe_filename(file.filename)}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        result = pipeline.classify_product(temp_path)
    except Exception as e:
        _os.remove(temp_path)
        raise HTTPException(status_code=400, detail=str(e))
    _os.remove(temp_path)
    return result

@app.get("/dashboard/stats")
def dashboard_stats(x_api_key: str = Header(None)):
    check_key(x_api_key)
    faces_dir = "app/models/faces"
    n_customers = len(os.listdir(faces_dir)) if os.path.exists(faces_dir) else 0
    return {
        "enrolled_customers": n_customers,
        "modules_active": ["face_recognition", "sentiment_analysis", "chatbot", "product_classifier"]
    }