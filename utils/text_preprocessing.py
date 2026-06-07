"""
Text Preprocessing Module
=========================
NLP preprocessing pipeline for resume and job description text.

Handles text cleaning, tokenization, stopword removal, and lemmatization
using NLTK. Includes automatic NLTK resource downloading with timeout
protection for offline environments.

Functions:
    clean_text(text: str) -> str
    preprocess_text(text: str) -> str
"""

import re
import socket
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def _initialize_nltk():
    """
    Download required NLTK resources if they are missing.

    Uses a 3-second socket timeout to prevent hanging in offline
    environments. Silently handles download failures.
    """
    original_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(3.0)

    resources = {
        "stopwords": "corpora/stopwords",
        "punkt": "tokenizers/punkt",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass  # Silently skip — fallback logic handles missing resources

    socket.setdefaulttimeout(original_timeout)


# Auto-initialize on import
_initialize_nltk()


def clean_text(text: str) -> str:
    """
    Clean raw text by removing noise and normalizing whitespace.

    Processing steps:
        1. Convert to lowercase
        2. Remove URLs (http/https/www)
        3. Remove email addresses
        4. Remove non-ASCII characters
        5. Remove special characters (keep alphanumeric, spaces, hyphens, periods)
        6. Collapse multiple whitespace characters

    Parameters
    ----------
    text : str
        Raw input text to clean.

    Returns
    -------
    str
        Cleaned text string, or empty string if input is falsy.
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9\s\-\.]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline: clean → tokenize → remove stopwords → lemmatize.

    Parameters
    ----------
    text : str
        Raw input text to preprocess.

    Returns
    -------
    str
        Space-separated string of preprocessed tokens. Returns empty
        string if input is empty or cleaning produces no content.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    # Tokenize with NLTK, fallback to whitespace split
    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        tokens = cleaned.split()

    # Load stopwords with fallback
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = set()

    lemmatizer = WordNetLemmatizer()

    processed_tokens = []
    for token in tokens:
        t = token.strip(".-")
        if t and t not in stop_words and len(t) > 1:
            try:
                lemma = lemmatizer.lemmatize(t)
            except Exception:
                lemma = t
            processed_tokens.append(lemma)

    return " ".join(processed_tokens)
