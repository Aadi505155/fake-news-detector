# Fake News Detector

Flask app serving a TF-IDF + Logistic Regression classifier.

## Run locally
    pip install -r requirements.txt
    python app.py            # http://localhost:5000

## Retrain (needs data/Fake.csv and data/True.csv)
    pip install -r requirements-train.txt
    python train.py          # writes model.joblib

## Deploy (Render / Railway / Heroku)
- Push this folder (including `model.joblib`) to GitHub. `data/` is git-ignored.
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
- Health check path: `/health`

If you retrain on a different scikit-learn version than the one in requirements.txt, re-run `train.py`
in the same environment you deploy with, otherwise the saved model may not load.
