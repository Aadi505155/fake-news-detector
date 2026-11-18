"""Shared text cleaning. Used by BOTH training and the web app so they never drift apart."""
import re

# "WASHINGTON (Reuters) - ", "LONDON (Reuters) - " etc. appear in ~99% of the real
# articles and 0% of the fake ones. If we leave them in, the model just learns
# "has Reuters tag = real" instead of learning anything about the writing.
_SOURCE_TAG = re.compile(r"^\W*[A-Z][A-Za-z .,/'’-]{0,60}\(\s*Reuters\s*\)\s*[-–—]?\s*")
_REUTERS_ANY = re.compile(r"\(?\s*reuters\s*\)?", re.IGNORECASE)
_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_ALPHA = re.compile(r"[^a-z]+")


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = _SOURCE_TAG.sub("", text.strip())
    text = _URL.sub(" ", text)
    text = _REUTERS_ANY.sub(" ", text)
    text = text.lower()
    return _NON_ALPHA.sub(" ", text).strip()
