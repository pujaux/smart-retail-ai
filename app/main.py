import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import shutil, os as _os

from app.services.face_recognition_module import recognize, enroll_customer
from app.services.sentiment_module import predict as predict_sentiment
from app.services.chatbot_module import Chatbot

app = FastAPI(title="Smart Retail AI Platform", version="1.0")

bot = Chatbot()
API_KEY = "retail-secret-key"  # simple demo key

def check_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

class SentimentRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    message: str

class EnrollRequest(BaseModel):
    name: str

# ---------- Endpoints ----------

@app.get("/")
def root():
    return {"status": "Smart Retail AI Platform is running"}

@app.post("/recognize-face")
async def recognize_face(file: UploadFile = File(...), x_api_key: str = Header(None)):
    check_key(x_api_key)
    temp_path = f"data/_upload_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = recognize(temp_path)
    _os.remove(temp_path)
    return result

@app.post("/enroll-customer")
async def enroll(name: str, file: UploadFile = File(...), x_api_key: str = Header(None)):
    check_key(x_api_key)
    temp_path = f"data/_upload_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    success = enroll_customer(name, temp_path)
    _os.remove(temp_path)
    if not success:
        raise HTTPException(status_code=400, detail="No face detected in image")
    return {"enrolled": True, "name": name}

@app.post("/analyze-sentiment")
def analyze_sentiment(req: SentimentRequest, x_api_key: str = Header(None)):
    check_key(x_api_key)
    return predict_sentiment(req.text)

@app.post("/chatbot")
def chatbot_reply(req: ChatRequest, x_api_key: str = Header(None)):
    check_key(x_api_key)
    return bot.get_response(req.message)

@app.get("/dashboard/stats")
def dashboard_stats(x_api_key: str = Header(None)):
    check_key(x_api_key)
    faces_dir = "app/models/faces"
    n_customers = len(os.listdir(faces_dir)) if os.path.exists(faces_dir) else 0
    return {
        "enrolled_customers": n_customers,
        "modules_active": ["face_recognition", "sentiment_analysis", "chatbot"]
    }
