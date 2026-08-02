"""
Unified pipeline — loads all models once so app/main.py doesn't need to
manage each service's initialization separately. Import `pipeline` from
here and call its methods from the API layer.
"""
from app.services.face_recognition_module import recognize as _recognize_face, enroll_customer as _enroll_customer
from app.services.sentiment_module import predict as _predict_sentiment, predict_distilbert as _predict_distilbert
from app.services.chatbot_module import Chatbot
from app.services.product_classifier import predict as _predict_product


class SmartRetailPipeline:
    """Loads all AI modules once at startup and exposes a single interface."""

    def __init__(self):
        print("[pipeline] Loading chatbot intents...")
        self.chatbot = Chatbot()
        print("[pipeline] All modules ready.")

    def recognize_face(self, image_path):
        return _recognize_face(image_path)

    def enroll_customer(self, name, image_path):
        return _enroll_customer(name, image_path)

    def analyze_sentiment(self, text, use_distilbert=False):
        if use_distilbert:
            return _predict_distilbert(text)
        return _predict_sentiment(text)

    def chat(self, message):
        return self.chatbot.get_response(message)

    def classify_product(self, image_path):
        return _predict_product(image_path)


# Singleton instance — created once when this module is first imported
pipeline = SmartRetailPipeline()
