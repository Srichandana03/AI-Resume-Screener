"""
Skill Extractor Module
======================
Extracts technical skills from text and performs skill gap analysis.

Contains a predefined database of 17 in-demand tech skills with curated
learning resource recommendations. Uses case-insensitive regex matching
with word boundaries for accurate skill detection.

Functions:
    extract_skills(text: str) -> list
    analyze_skill_gap(candidate_skills: list, job_skills: list) -> dict

Constants:
    SKILLS_DB : list[str]           — Predefined skill names
    SKILL_RECOMMENDATIONS : dict    — Skill → learning resource mapping
"""

import re

# ─── Predefined Skill Database ───────────────────────────────────────────────
SKILLS_DB = [
    "Python", "Java", "SQL", "Machine Learning", "Deep Learning", "NLP",
    "TensorFlow", "PyTorch", "Scikit-Learn", "FastAPI", "Docker", "AWS",
    "Git", "Data Structures", "Algorithms", "React", "JavaScript",
]

# ─── Curated Learning Recommendations ────────────────────────────────────────
SKILL_RECOMMENDATIONS = {
    "Python": "Python for Everybody Specialization (Coursera) / Official Python Docs",
    "Java": "Java Programming Masterclass (Udemy) / University of Helsinki Java MOOC",
    "SQL": "Complete SQL Bootcamp (Udemy) / Khan Academy SQL Courses",
    "Machine Learning": "Machine Learning Specialization by Andrew Ng (Coursera)",
    "Deep Learning": "Deep Learning Specialization by Andrew Ng (Coursera) / Fast.ai",
    "NLP": "Natural Language Processing Specialization (DeepLearning.AI)",
    "TensorFlow": "TensorFlow Developer Professional Certificate (Coursera)",
    "PyTorch": "PyTorch for Deep Learning Bootcamp / PyTorch Official Tutorials",
    "Scikit-Learn": "Hands-On ML with Scikit-Learn, Keras, and TensorFlow (O'Reilly)",
    "FastAPI": "FastAPI Official Tutorial and Docs / Building APIs with FastAPI (TestDriven.io)",
    "Docker": "Docker & Kubernetes: The Complete Guide (Udemy) / Docker Docs",
    "AWS": "AWS Certified Cloud Practitioner / AWS Skill Builder Learning Paths",
    "Git": "Git & GitHub: The Complete Git Guide / Git Immersion",
    "Data Structures": "Data Structures & Algorithms Specialization (UC San Diego, Coursera)",
    "Algorithms": "Algorithms, Part I & II (Princeton, Coursera) / LeetCode Exercises",
    "React": "React — The Complete Guide (Udemy) / Official React Documentation (react.dev)",
    "JavaScript": "Eloquent JavaScript (Book) / The Modern JavaScript Tutorial (javascript.info)",
}


def extract_skills(text: str) -> list:
    """
    Extract technical skills present in the given text.

    Performs case-insensitive regex matching with word boundaries against
    the predefined SKILLS_DB. Includes special handling for multi-word
    and hyphenated skill names (e.g., "Scikit-Learn", "Data Structures").

    Parameters
    ----------
    text : str
        Input text to search for skills (resume or job description).

    Returns
    -------
    list[str]
        List of matched skill names from SKILLS_DB.
    """
    if not text:
        return []

    text_lower = text.lower()
    found_skills = []

    for skill in SKILLS_DB:
        # Build regex pattern with word boundaries
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"

        # Custom patterns for tricky skill names
        if skill.lower() == "scikit-learn":
            pattern = r"\bscikit[- ]learn\b"
        elif skill.lower() == "data structures":
            pattern = r"\bdata\s+structures?\b"

        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return found_skills


def analyze_skill_gap(candidate_skills: list, job_skills: list) -> dict:
    """
    Compare candidate skills against job requirements to identify gaps.

    Parameters
    ----------
    candidate_skills : list[str]
        Skills extracted from the candidate's resume.
    job_skills : list[str]
        Skills extracted from the job description.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``matching`` (list[str]): Skills present in both sets (sorted)
        - ``missing`` (list[str]): Skills required but absent (sorted)
        - ``recommendations`` (dict): Missing skill → learning resource
    """
    candidate_set = set(candidate_skills)
    job_set = set(job_skills)

    matching = sorted(candidate_set.intersection(job_set))
    missing = sorted(job_set.difference(candidate_set))

    recommendations = {
        skill: SKILL_RECOMMENDATIONS.get(skill, "Online tutorials and official documentation.")
        for skill in missing
    }

    return {
        "matching": matching,
        "missing": missing,
        "recommendations": recommendations,
    }
