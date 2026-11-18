import os

import joblib
from flask import Flask, jsonify, render_template, request

from preprocess import clean_text  # must be importable: joblib model was trained with this cleaning

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

app = Flask(__name__)
model = joblib.load(MODEL_PATH)  # loaded once at startup, not on every request


def predict(text: str) -> dict:
    cleaned = clean_text(text)
    if len(cleaned.split()) < 5:
        raise ValueError("Please enter at least a few words of the article.")
    proba_real = float(model.predict_proba([cleaned])[0][1])
    is_real = proba_real >= 0.5
    return {
        "label": "Real" if is_real else "Fake",
        "confidence": round((proba_real if is_real else 1 - proba_real) * 100, 1),
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(silent=True) or request.form
    text = (data.get("text") or "").strip()
    try:
        return jsonify(predict(text))
    except ValueError as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
