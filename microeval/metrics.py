import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import nltk
import numpy as np

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Global cache for GPT-2 tokenizer and model
_tokenizer = None
_model = None

def _load_gpt2():
    """
    Load and cache GPT-2 tokenizer and model.

    Returns:
        tuple: (GPT2TokenizerFast, GPT2LMHeadModel)
    """
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        _model = GPT2LMHeadModel.from_pretrained("gpt2")
        _model.eval()
    return _tokenizer, _model

def perplexity(text: str) -> float:
    """
    Compute the perplexity of a text using GPT-2.

    Args:
        text (str): Input text.

    Returns:
        float: Perplexity score.
    """
    tokenizer, model = _load_gpt2()
    encodings = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

def flesch_kincaid(text: str) -> float:
    """
    Compute the Flesch-Kincaid readability score for a given text.

    Args:
        text (str): Input text.

    Returns:
        float: Flesch-Kincaid readability score.
    """
    from nltk.tokenize import sent_tokenize, word_tokenize

    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    syllables = sum(_count_syllables(w) for w in words)

    num_sentences = max(1, len(sentences))
    num_words = max(1, len(words))

    return 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (syllables / num_words)

def _count_syllables(word: str) -> int:
    """
    Estimate the number of syllables in a word.

    Args:
        word (str): Input word.

    Returns:
        int: Estimated syllable count.
    """
    vowels = "aeiouy"
    word = word.lower().strip(".:;?!")
    if not word:
        return 0
    count = 0
    prev_char_was_vowel = False
    for char in word:
        if char in vowels:
            if not prev_char_was_vowel:
                count += 1
            prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False
    if word.endswith("e"):
        count = max(1, count - 1)
    return max(1, count)
