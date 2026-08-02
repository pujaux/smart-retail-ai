import pandas as pd
import pickle, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

MODEL_PATH = "app/models/sentiment_model.pkl"
VEC_PATH = "app/models/sentiment_vectorizer.pkl"

# ---------- DistilBERT (optional upgrade) ----------
_distilbert_pipeline = None

def _get_distilbert():
    global _distilbert_pipeline
    if _distilbert_pipeline is None:
        from transformers import pipeline
        _distilbert_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _distilbert_pipeline

def predict_distilbert(text):
    """Uses a pretrained DistilBERT model fine-tuned for sentiment (binary: POSITIVE/NEGATIVE).
    Downloads the model on first use (~268MB) and caches it locally afterward."""
    clf = _get_distilbert()
    result = clf(text)[0]
    label = "positive" if result["label"] == "POSITIVE" else "negative"
    return {"text": text, "sentiment": label, "confidence": round(float(result["score"]), 3), "model": "distilbert"}
# ---------------------------------------------------

def train(csv_path="data/reviews.csv", text_col="review", label_col="sentiment"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, label_col])

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[label_col], test_size=0.2, random_state=42
    )

    vec = TfidfVectorizer(max_features=3000, stop_words="english")
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    acc = model.score(X_test_vec, y_test)
    print(f"Test accuracy: {acc:.3f}")

    os.makedirs("app/models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VEC_PATH, "wb") as f:
        pickle.dump(vec, f)
    return acc

def predict(text):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        vec = pickle.load(f)
    X = vec.transform([text])
    pred = model.predict(X)[0]
    proba = max(model.predict_proba(X)[0])
    return {"text": text, "sentiment": pred, "confidence": round(float(proba), 3)}

if __name__ == "__main__":
    train()
    print(predict("This product is amazing, I love it!"))
    print(predict("Terrible quality, very disappointed."))

    # Uncomment to test DistilBERT (downloads ~268MB on first run):
    # print(predict_distilbert("This product is amazing, I love it!"))
    # print(predict_distilbert("Terrible quality, very disappointed."))
