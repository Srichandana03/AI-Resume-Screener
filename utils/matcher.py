"""
Matcher Module
==============
Computes hybrid matching scores between resume text and job descriptions.

Combines three signals into a weighted 0–100 match score:
    - TF-IDF Cosine Similarity (50% weight)
    - Skill Overlap Ratio (40% weight)
    - Experience Fit (10% weight)

Functions:
    extract_experience_years(text: str) -> int
    calculate_match_metrics(resume_text: str, jd_text: str) -> dict
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.text_preprocessing import preprocess_text
from utils.skill_extractor import extract_skills


def extract_experience_years(text: str) -> int:
    """
    Extract years of experience from text using multiple regex patterns.

    Searches for patterns like "5 years of experience", "3+ yrs", etc.
    Filters out numbers that are likely years (e.g., 2020) or irrelevant.
    Falls back to keyword heuristics ("senior" → 5, "intern" → 1).

    Parameters
    ----------
    text : str
        Input text to extract experience from.

    Returns
    -------
    int
        Maximum experience years found, or 0 if none detected.
    """
    if not text:
        return 0

    patterns = [
        r"\b(\d+)\+?\s*years?\s+(?:of\s+)?experience\b",
        r"\b(\d+)\+?\s*years?\b",
        r"\b(\d+)\+?\s*yrs?\b",
    ]

    years_found = []
    text_lower = text.lower()

    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for m in matches:
            try:
                val = int(m)
                # Filter out calendar years and zip codes
                if val < 30:
                    years_found.append(val)
            except ValueError:
                pass

    if years_found:
        return max(years_found)

    # Keyword-based fallback
    if any(kw in text_lower for kw in ("senior", "lead", "principal")):
        return 5
    elif any(kw in text_lower for kw in ("intern", "junior", "entry level")):
        return 1

    return 0


def calculate_match_metrics(resume_text: str, jd_text: str) -> dict:
    """
    Compute comprehensive matching metrics between a resume and job description.

    Calculates TF-IDF cosine similarity, skill overlap ratio, experience
    comparison, and a weighted hybrid match score (0–100).

    Score Weights:
        - Cosine Similarity: 50%
        - Skill Overlap Ratio: 40%
        - Experience Fit: 10%

    Parameters
    ----------
    resume_text : str
        Raw text extracted from the candidate's resume.
    jd_text : str
        Raw text of the job description.

    Returns
    -------
    dict
        Dictionary containing:
        - ``tfidf_similarity`` (float): Cosine similarity as percentage
        - ``skill_overlap_ratio`` (float): Skill match as percentage
        - ``resume_skills`` (list): Skills found in resume
        - ``jd_skills`` (list): Skills found in JD
        - ``matched_skills`` (list): Skills in both
        - ``missing_skills`` (list): Skills in JD but not resume
        - ``resume_experience`` (int): Candidate's experience years
        - ``jd_experience`` (int): Required experience years
        - ``match_score`` (float): Final hybrid score (0–100)
    """
    # ── TF-IDF Cosine Similarity ─────────────────────────
    resume_clean = preprocess_text(resume_text)
    jd_clean = preprocess_text(jd_text)

    if not resume_clean or not jd_clean:
        tfidf_similarity = 0.0
    else:
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([jd_clean, resume_clean])
            tfidf_similarity = float(
                cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            )
        except Exception:
            tfidf_similarity = 0.0

    # ── Skill Extraction ─────────────────────────────────
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    if not jd_skills:
        skill_overlap_ratio = 1.0
        matched_skills = []
        missing_skills = []
    else:
        matched_skills = sorted(set(resume_skills).intersection(set(jd_skills)))
        missing_skills = sorted(set(jd_skills).difference(set(resume_skills)))
        skill_overlap_ratio = len(matched_skills) / len(jd_skills)

    # ── Experience Comparison ────────────────────────────
    resume_exp = extract_experience_years(resume_text)
    jd_exp = extract_experience_years(jd_text)

    if jd_exp == 0:
        exp_score = 1.0
    else:
        exp_score = min(resume_exp / jd_exp, 1.0)

    # ── Hybrid Match Score (0–100) ───────────────────────
    hybrid_score = (
        tfidf_similarity * 0.5
        + skill_overlap_ratio * 0.4
        + exp_score * 0.1
    )
    match_score_100 = round(hybrid_score * 100, 1)

    return {
        "tfidf_similarity": round(tfidf_similarity * 100, 1),
        "skill_overlap_ratio": round(skill_overlap_ratio * 100, 1),
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "resume_experience": resume_exp,
        "jd_experience": jd_exp,
        "match_score": match_score_100,
    }
