import math
import re
from typing import List
import nltk

# Ensure punkt available - user should have downloaded, but be tolerant
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass

def tokenize(text: str) -> List[str]:
    """Simple word tokenizer using nltk.sent_tokenize + split."""
    if text is None:
        return []
    # Remove weird whitespace
    text = text.strip()
    # split sentences then words
    words = []
    try:
        sents = nltk.sent_tokenize(text)
        for s in sents:
            # simple word tokenization: split on non-word but keep apostrophes
            tokens = re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']", s)
            words.extend([t.lower() for t in tokens if t.strip()])
    except Exception:
        # fallback
        tokens = re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']", text)
        words = [t.lower() for t in tokens if t.strip()]
    return words

def ngrams(tokens, n):
    if n <= 0:
        return []
    return [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]

def safe_div(a, b):
    return a / b if b else 0.0

def flatten(list_of_lists):
    return [x for lst in list_of_lists for x in lst]
