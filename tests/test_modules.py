"""
Automated tests for the Smart Retail AI Platform.
Run with: pytest tests/ -v
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.services.sentiment_module import predict as predict_sentiment
from app.services.chatbot_module import Chatbot
from app.services.product_classifier import predict as predict_product


class TestSentimentModule:
    def test_positive_review(self):
        result = predict_sentiment("I absolutely love this product, amazing quality!")
        assert result["sentiment"] == "positive"

    def test_negative_review(self):
        result = predict_sentiment("Terrible quality, broke immediately, very disappointed.")
        assert result["sentiment"] == "negative"

    def test_confidence_in_range(self):
        result = predict_sentiment("It's an okay product.")
        assert 0.0 <= result["confidence"] <= 1.0


class TestChatbotModule:
    @pytest.fixture
    def bot(self):
        return Chatbot()

    def test_greeting_intent(self, bot):
        result = bot.get_response("hi there")
        assert result["tag"] == "greeting"

    def test_order_status_intent(self, bot):
        result = bot.get_response("where is my order")
        assert result["tag"] == "order_status"

    def test_unknown_message(self, bot):
        result = bot.get_response("asdkjaskjdlaksjd")
        assert result["tag"] == "unknown"


class TestProductClassifier:
    def test_predicts_known_category(self):
        sample = "data/products/shoes/shoes_0.jpg"
        if os.path.exists(sample):
            result = predict_product(sample)
            assert result["category"] in ["shoes", "electronics", "clothing"]
            assert 0.0 <= result["confidence"] <= 1.0
