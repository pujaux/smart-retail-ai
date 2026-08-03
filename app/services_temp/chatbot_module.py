import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INTENTS_PATH = "data/intents.json"

class Chatbot:
    def __init__(self, path=INTENTS_PATH):
        with open(path) as f:
            data = json.load(f)
        self.intents = data["intents"]

        self.patterns, self.tags = [], []
        for intent in self.intents:
            for p in intent["patterns"]:
                self.patterns.append(p.lower())
                self.tags.append(intent["tag"])

        self.vec = TfidfVectorizer()
        self.pattern_vecs = self.vec.fit_transform(self.patterns)

    def get_response(self, message, threshold=0.3):
        msg_vec = self.vec.transform([message.lower()])
        sims = cosine_similarity(msg_vec, self.pattern_vecs)[0]
        best_idx = sims.argmax()

        if sims[best_idx] < threshold:
            return {"tag": "unknown", "response": "Sorry, I didn't understand that. Could you rephrase?", "confidence": float(sims[best_idx])}

        tag = self.tags[best_idx]
        for intent in self.intents:
            if intent["tag"] == tag:
                return {"tag": tag, "response": random.choice(intent["responses"]), "confidence": float(sims[best_idx])}

if __name__ == "__main__":
    bot = Chatbot()
    for msg in ["hi there", "where is my order", "do you have discounts", "asdkjaskjd"]:
        print(msg, "->", bot.get_response(msg))
