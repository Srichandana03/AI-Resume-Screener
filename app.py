"""
AI Resume Screening & Skill Gap Analyzer
========================================
Main Streamlit dashboard application. Provides a professional recruiter dashboard
with multiple sections including Job Description configuration, candidate PDF resume
upload, leaderboard ranking, skill gap analysis, and interactive ML analytics
with SHAP explainability.

Author: GitHub Portfolio Project
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Utilities
from utils.pdf_parser import extract_text_from_pdf
from utils.text_preprocessing import preprocess_text
from utils.skill_extractor import extract_skills, analyze_skill_gap
from utils.matcher import calculate_match_metrics, extract_experience_years
from utils.ml_model import load_best_model, get_shap_waterfall_plot, FEATURE_COLUMNS

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screening & Skill Gap Analyzer",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS Theme ───────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font Import ───────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Root Variables ───────────────────────────────────── */
    :root {
        --bg-primary: #0F172A;
        --bg-card: #1E293B;
        --bg-card-hover: #263548;
        --border-subtle: #334155;
        --border-accent: #3B82F6;
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --accent-blue: #3B82F6;
        --accent-emerald: #10B981;
        --accent-violet: #8B5CF6;
        --accent-amber: #F59E0B;
        --accent-rose: #F43F5E;
        --gradient-primary: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 50%, #10B981 100%);
        --gradient-warm: linear-gradient(135deg, #F59E0B 0%, #F43F5E 100%);
        --shadow-card: 0 4px 24px rgba(0,0,0,0.25);
        --shadow-glow: 0 0 30px rgba(59,130,246,0.15);
        --radius-lg: 16px;
        --radius-md: 12px;
        --radius-sm: 8px;
        --radius-full: 9999px;
    }

    /* ── Global Styles ────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary);
    }

    .stApp {
        background: var(--bg-primary);
    }

    /* ── Sidebar ──────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
        border-right: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem !important;
        padding: 0.45rem 0;
    }

    /* ── Headings ─────────────────────────────────────────── */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }
    .subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
        font-weight: 400;
        line-height: 1.6;
    }
    .section-heading {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Premium Metric Cards ─────────────────────────────── */
    .metric-card {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 1.75rem;
        border: 1px solid var(--border-subtle);
        box-shadow: var(--shadow-card);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: var(--border-accent);
        box-shadow: var(--shadow-glow);
    }
    .metric-card:hover::before {
        opacity: 1;
    }
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .metric-title {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: var(--text-primary);
        margin-top: 0.4rem;
        letter-spacing: -0.02em;
    }
    .metric-desc {
        font-size: 0.8rem;
        color: var(--accent-emerald);
        margin-top: 0.5rem;
        font-weight: 500;
    }

    /* ── Feature Cards ────────────────────────────────────── */
    .feature-card {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        border: 1px solid var(--border-subtle);
        box-shadow: var(--shadow-card);
        transition: all 0.3s ease;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        border-color: var(--accent-violet);
    }
    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 0.75rem;
    }
    .feature-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.55;
    }

    /* ── Skill Badges ─────────────────────────────────────── */
    .skill-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: var(--bg-card);
        color: var(--text-primary);
        padding: 0.35rem 0.85rem;
        border-radius: var(--radius-full);
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem;
        border: 1px solid var(--border-subtle);
        transition: all 0.2s ease;
    }
    .skill-badge:hover { transform: scale(1.05); }
    .skill-badge-match {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border-color: rgba(16, 185, 129, 0.35);
    }
    .skill-badge-missing {
        background: rgba(244, 63, 94, 0.12);
        color: #FB7185;
        border-color: rgba(244, 63, 94, 0.35);
    }
    .skill-badge-neutral {
        background: rgba(139, 92, 246, 0.12);
        color: #A78BFA;
        border-color: rgba(139, 92, 246, 0.35);
    }

    /* ── Status Badges ────────────────────────────────────── */
    .status-shortlisted {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        padding: 0.3rem 0.8rem;
        border-radius: var(--radius-full);
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-rejected {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(244, 63, 94, 0.15);
        color: #FB7185;
        padding: 0.3rem 0.8rem;
        border-radius: var(--radius-full);
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }

    /* ── Upload Area ──────────────────────────────────────── */
    .upload-area {
        background: var(--bg-card);
        border: 2px dashed var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 2.5rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    .upload-area:hover {
        border-color: var(--accent-blue);
        background: rgba(59, 130, 246, 0.05);
    }
    .upload-icon { font-size: 3rem; margin-bottom: 0.75rem; }
    .upload-text {
        font-size: 1rem;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }
    .upload-hint {
        font-size: 0.8rem;
        color: var(--text-muted);
    }

    /* ── Info Cards ────────────────────────────────────────── */
    .info-panel {
        background: var(--bg-card);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        border: 1px solid var(--border-subtle);
        margin: 0.75rem 0;
    }

    /* ── Architecture Diagram ─────────────────────────────── */
    .arch-step {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        padding: 1rem 1.25rem;
        background: var(--bg-card);
        border-radius: var(--radius-md);
        border: 1px solid var(--border-subtle);
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }
    .arch-step:hover {
        border-color: var(--accent-blue);
        transform: translateX(4px);
    }
    .arch-num {
        flex-shrink: 0;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: var(--gradient-primary);
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .arch-content { flex: 1; }
    .arch-title {
        font-weight: 700;
        color: var(--text-primary);
        font-size: 0.95rem;
    }
    .arch-desc {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 0.2rem;
        line-height: 1.5;
    }

    /* ── Learning Roadmap ─────────────────────────────────── */
    .roadmap-item {
        background: var(--bg-card);
        border-radius: var(--radius-md);
        padding: 1.1rem 1.25rem;
        border-left: 3px solid var(--accent-violet);
        margin-bottom: 0.65rem;
        transition: all 0.2s ease;
    }
    .roadmap-item:hover {
        border-left-color: var(--accent-blue);
        transform: translateX(4px);
    }
    .roadmap-skill {
        font-weight: 700;
        color: var(--accent-violet);
        font-size: 0.95rem;
    }
    .roadmap-resource {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 0.25rem;
        line-height: 1.5;
    }

    /* ── Score Ring ────────────────────────────────────────── */
    .score-ring {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .score-value-large {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }
    .score-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }

    /* ── Divider ──────────────────────────────────────────── */
    .styled-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
        margin: 2rem 0;
    }

    /* ── Hide Streamlit Branding ──────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Scrollbar ─────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border-subtle); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

    /* ── Streamlit Component Overrides ─────────────────────── */
    .stDataFrame { border-radius: var(--radius-md) !important; overflow: hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm) !important;
        padding: 0.5rem 1.25rem !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────────────────────
if "job_description" not in st.session_state:
    st.session_state.job_description = ""
if "jd_skills" not in st.session_state:
    st.session_state.jd_skills = []
if "jd_experience" not in st.session_state:
    st.session_state.jd_experience = 0
if "processed_resumes" not in st.session_state:
    st.session_state.processed_resumes = {}


# ─── Load ML Model (with error handling) ─────────────────────────────────────
@st.cache_resource
def _load_model():
    try:
        return load_best_model("models")
    except Exception:
        return None, "Unavailable"

model, model_name = _load_model()

@st.cache_data
def _load_train_data():
    path = "models/X_train.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

X_train_df = _load_train_data()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='text-align:center; padding:1.25rem 0 0.5rem;'>"
    "<span style='font-size:2.2rem;'>💼</span><br>"
    "<span style='font-size:1.3rem; font-weight:800; "
    "background:linear-gradient(135deg,#3B82F6,#8B5CF6);-webkit-background-clip:text;"
    "-webkit-text-fill-color:transparent;'>AI Recruiter</span>"
    "</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("<hr style='border-color:#334155; margin:0.75rem 0;'>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠  Home",
        "📋  Job Description",
        "📄  Resume Upload",
        "🏆  Resume Ranking",
        "🔍  Skill Gap Analysis",
        "📊  Analytics Dashboard",
    ],
)

st.sidebar.markdown("<hr style='border-color:#334155; margin:0.75rem 0;'>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div style='padding:0.5rem;'>"
    "<p style='font-size:0.8rem;color:#64748B;text-transform:uppercase;font-weight:700;"
    "letter-spacing:0.08em;margin-bottom:0.75rem;'>System Status</p>"
    f"<p style='font-size:0.85rem;color:#94A3B8;margin:0.4rem 0;'>🤖 <b>ML Model:</b> {model_name}</p>"
    "<p style='font-size:0.85rem;color:#94A3B8;margin:0.4rem 0;'>🛠 <b>Skill Database:</b> 17 Tech Skills</p>"
    "<p style='font-size:0.85rem;color:#34D399;margin:0.4rem 0;'>✅ <b>Project Status:</b> Running Successfully</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

def render_metric_card(icon: str, title: str, value: str, description: str = ""):
    """
    Render a premium glassmorphic metric card using HTML/CSS.

    Parameters
    ----------
    icon : str
        Emoji or icon character to display in the card.
    title : str
        Upper-case muted title text.
    value : str
        Bold main numeric or text value.
    description : str, optional
        Sub-value description, typically colored positive emerald.
    """
    desc_html = f"<div class='metric-desc'>{description}</div>" if description else ""
    st.markdown(
        f"""<div class='metric-card'>
            <div class='metric-icon'>{icon}</div>
            <div class='metric-title'>{title}</div>
            <div class='metric-value'>{value}</div>
            {desc_html}
        </div>""",
        unsafe_allow_html=True,
    )

def score_color(score: float) -> str:
    """
    Map match scores to semantic theme colors.

    Parameters
    ----------
    score : float
        Match score value (0 to 100).

    Returns
    -------
    str
        Hex color string representing green (high), amber (medium), or rose (low).
    """
    if score >= 70:
        return "#34D399"
    elif score >= 40:
        return "#FBBF24"
    return "#FB7185"

def render_page_header(title: str, subtitle: str):
    """
    Render the standard header with gradient main title and description.

    Parameters
    ----------
    title : str
        Main page title.
    subtitle : str
        Description or subtitle text.
    """
    st.markdown(f"<h1 class='main-title'>{title}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='subtitle'>{subtitle}</p>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    render_page_header(
        "AI-Powered Resume Screening & Skill Gap Analyzer",
        "Automate candidate pre-screening, ranking, and skill gap identification with NLP & Machine Learning"
    )

    # ── Hero metrics row ─────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("🧠", "NLP Engine", "TF-IDF", "Tokenized & lemmatized matching")
    with col2:
        render_metric_card("🤖", "ML Prediction", model_name, "Shortlist probability scoring")
    with col3:
        render_metric_card("🎯", "Skill Database", "17 Skills", "Python, AWS, React & more")
    with col4:
        render_metric_card("📊", "Explainability", "SHAP", "Feature-level explanations")

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

    # ── Project Overview ─────────────────────────────────────
    st.markdown("<div class='section-heading'>🌟 Project Overview</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='info-panel' style='font-size:0.95rem;color:#CBD5E1;line-height:1.7;'>"
        "This application is designed for <b>HR recruiters</b> to automate candidate pre-screening and "
        "identify skill gaps for job openings. It eliminates manual resume review by parsing PDF resumes, "
        "extracting technical skills & experience, calculating matching similarity, and predicting "
        "shortlist feasibility using Machine Learning classifiers — all explained through SHAP."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

    # ── Architecture Pipeline ────────────────────────────────
    st.markdown("<div class='section-heading'>🏗️ Architecture & Pipeline</div>", unsafe_allow_html=True)

    pipeline_steps = [
        ("PDF Parsing", "Upload candidate resumes in PDF format. Text is extracted using <b>pypdf</b> for downstream analysis."),
        ("NLP Preprocessing", "Raw text is normalized, tokenized, stopword-filtered, and lemmatized using <b>NLTK</b>."),
        ("Skill & Experience Extraction", "Regex and word-boundary matching isolates <b>17 technical skills</b> and years of experience."),
        ("Hybrid Score Matching", "<b>TF-IDF + Cosine Similarity</b> (50%) combined with skill overlap (40%) and experience fit (10%)."),
        ("ML Classification", "Logistic Regression, Random Forest, and XGBoost predict <b>shortlist probability</b> on extracted features."),
        ("SHAP Explainability", "Feature importance is visualized per candidate — showing which factors drive the prediction."),
    ]
    for i, (title, desc) in enumerate(pipeline_steps, 1):
        st.markdown(
            f"""<div class='arch-step'>
                <div class='arch-num'>{i}</div>
                <div class='arch-content'>
                    <div class='arch-title'>{title}</div>
                    <div class='arch-desc'>{desc}</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

    # ── Key Features ─────────────────────────────────────────
    st.markdown("<div class='section-heading'>✨ Key Features</div>", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    features = [
        ("📄", "Smart PDF Parsing", "Automatically extracts and cleans text from multi-page PDF resumes with error-tolerant processing."),
        ("🎯", "Hybrid Scoring", "Combines TF-IDF similarity, skill overlap ratio, and experience fit into a single 0–100 match score."),
        ("🔬", "Skill Gap Analysis", "Identifies missing skills and recommends curated online courses and certifications for improvement."),
        ("🏆", "Candidate Leaderboard", "Ranks all candidates in a sortable table with match scores, skill counts, and ML predictions."),
        ("📊", "Interactive Analytics", "Plotly-powered dashboards with bar charts, scatter plots, pie charts, and model comparisons."),
        ("🧪", "SHAP Explainability", "Transparent ML predictions — see exactly which features push each candidate toward or against shortlisting."),
    ]
    for col, (icon, title, desc) in zip([fc1, fc2, fc3, fc1, fc2, fc3], features):
        with col:
            st.markdown(
                f"""<div class='feature-card'>
                    <div class='feature-icon'>{icon}</div>
                    <div class='feature-title'>{title}</div>
                    <div class='feature-desc'>{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

    # ── Quick Start ──────────────────────────────────────────
    st.markdown("<div class='section-heading'>🚀 Quick Start Guide</div>", unsafe_allow_html=True)

    quick_steps = [
        ("📋", "Navigate to **Job Description** to paste a job description or choose a pre-loaded template."),
        ("📄", "Go to **Resume Upload** to upload candidate PDFs or select the included mock resumes."),
        ("🏆", "View the **Resume Ranking** leaderboard for a scored comparison of all candidates."),
        ("🔍", "Explore **Skill Gap Analysis** for per-candidate breakdowns with learning recommendations."),
        ("📊", "Check the **Analytics Dashboard** for interactive charts, model metrics, and SHAP explainability."),
    ]
    for icon, text in quick_steps:
        st.markdown(f"{icon} {text}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — JOB DESCRIPTION INPUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋  Job Description":
    render_page_header(
        "Job Description Input",
        "Configure the targeting criteria for candidate screening"
    )

    st.markdown("<div class='section-heading'>📝 Choose a template or paste custom text</div>", unsafe_allow_html=True)

    template = st.selectbox(
        "Load Job Description Template:",
        ["Custom (Write Below)", "Machine Learning Engineer", "Full-Stack Web Developer"],
    )

    default_text = ""
    if template == "Machine Learning Engineer" and os.path.exists("data/jd_ml_engineer.txt"):
        with open("data/jd_ml_engineer.txt", "r") as f:
            default_text = f.read()
    elif template == "Full-Stack Web Developer" and os.path.exists("data/jd_web_developer.txt"):
        with open("data/jd_web_developer.txt", "r") as f:
            default_text = f.read()

    jd_input = st.text_area("Job Description Content:", value=default_text, height=250)

    if st.button("🔍  Analyze & Save Criteria", use_container_width=True):
        if jd_input.strip() == "":
            st.error("⚠️ Please enter a valid job description!")
        else:
            with st.spinner("Analyzing job description..."):
                st.session_state.job_description = jd_input
                skills = extract_skills(jd_input)
                exp = extract_experience_years(jd_input)
                st.session_state.jd_skills = skills
                st.session_state.jd_experience = exp

            st.success("✅ Job description successfully analyzed and saved!")

    if st.session_state.job_description:
        st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-heading'>📋 Analyzed Criteria</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            render_metric_card("📅", "Required Experience", f"{st.session_state.jd_experience} years", "Minimum experience level")
        with col2:
            render_metric_card("🎯", "Skills Identified", f"{len(st.session_state.jd_skills)}", "Technical skills detected")

        if st.session_state.jd_skills:
            st.markdown("<div class='section-heading' style='margin-top:1.5rem;'>🛠 Target Technical Skills</div>", unsafe_allow_html=True)
            badges = " ".join(
                f"<span class='skill-badge skill-badge-match'>✔ {s}</span>"
                for s in st.session_state.jd_skills
            )
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.info("No predefined technical skills were identified in the job description.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — RESUME UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📄  Resume Upload":
    render_page_header(
        "Resume Upload & Processing",
        "Upload candidate PDF resumes for automated parsing and screening"
    )

    if not st.session_state.job_description:
        st.warning("⚠️ Please configure and save a Job Description first before uploading resumes.")
        st.stop()

    # ── Option A: Mock resumes ───────────────────────────────
    st.markdown("<div class='section-heading'>📂 Select Sample Resumes</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.88rem;color:#94A3B8;margin-bottom:0.75rem;'>"
        "Pre-generated mock candidate resumes are available for immediate testing.</div>",
        unsafe_allow_html=True,
    )

    mock_files = []
    if os.path.exists("resumes"):
        mock_files = sorted([f for f in os.listdir("resumes") if f.endswith(".pdf")])

    selected_mocks = st.multiselect("Select mock resume files:", mock_files)

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

    # ── Option B: Upload ─────────────────────────────────────
    st.markdown("<div class='section-heading'>📤 Upload PDF Resumes</div>", unsafe_allow_html=True)
    st.markdown(
        """<div class='upload-area'>
            <div class='upload-icon'>📎</div>
            <div class='upload-text'>Drag and drop your PDF resume files here</div>
            <div class='upload-hint'>Supported format: PDF • Multiple files accepted</div>
        </div>""",
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Upload PDF Resumes:",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Show uploaded file names
    if uploaded_files:
        st.markdown("<div class='section-heading' style='font-size:1rem;'>📋 Uploaded Files</div>", unsafe_allow_html=True)
        for uf in uploaded_files:
            st.markdown(
                f"<span class='skill-badge skill-badge-neutral'>📄 {uf.name}</span>",
                unsafe_allow_html=True,
            )

    st.markdown("")  # spacing

    if st.button("🚀  Process & Screen Candidates", use_container_width=True):
        candidates_to_process = []

        for m in selected_mocks:
            path = os.path.join("resumes", m)
            display_name = m.replace(".pdf", "").replace("_", " ").title()
            candidates_to_process.append((display_name, path, False))

        for u in uploaded_files:
            display_name = u.name.replace(".pdf", "").replace("_", " ").title()
            candidates_to_process.append((display_name, u, True))

        if not candidates_to_process:
            st.error("⚠️ Please upload or select at least one resume!")
        else:
            progress_bar = st.progress(0, text="Initializing screening pipeline...")
            results_container = st.container()

            for idx, (name, source, is_upload) in enumerate(candidates_to_process):
                progress_bar.progress(
                    (idx) / len(candidates_to_process),
                    text=f"Processing candidate: {name}...",
                )

                try:
                    if is_upload:
                        text = extract_text_from_pdf(source)
                    else:
                        with open(source, "rb") as f:
                            text = extract_text_from_pdf(f)

                    if not text.strip():
                        with results_container:
                            st.warning(f"⚠️ Could not extract text from **{name}** — the file may be empty or scanned.")
                        continue

                    metrics = calculate_match_metrics(text, st.session_state.job_description)

                    if model is not None:
                        try:
                            features = [
                                metrics["match_score"],
                                len(metrics["matched_skills"]),
                                len(metrics["missing_skills"]),
                                metrics["resume_experience"],
                            ]
                            prob = float(model.predict_proba([features])[0][1])
                            pred = int(model.predict([features])[0])
                        except Exception:
                            prob, pred = 0.0, 0
                    else:
                        prob, pred = 0.0, 0

                    st.session_state.processed_resumes[name] = {
                        "text": text,
                        "metrics": metrics,
                        "ml_prediction": pred,
                        "ml_probability": prob,
                    }

                except Exception as e:
                    with results_container:
                        st.error(f"❌ Error processing **{name}**: {e}")

                progress_bar.progress((idx + 1) / len(candidates_to_process))

            progress_bar.progress(1.0, text="✅ Screening complete!")

            with results_container:
                total = len(st.session_state.processed_resumes)
                st.success(f"🎉 Successfully screened **{total}** candidate resume{'s' if total != 1 else ''}!")

                # Show preview of extracted text
                st.markdown("<div class='section-heading' style='margin-top:1rem;'>📝 Extracted Text Preview</div>", unsafe_allow_html=True)
                for name, data in list(st.session_state.processed_resumes.items())[-len(candidates_to_process):]:
                    with st.expander(f"📄 {name}", expanded=False):
                        st.text(data["text"][:1000] + ("..." if len(data["text"]) > 1000 else ""))


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — RESUME RANKING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆  Resume Ranking":
    render_page_header(
        "Candidate Screening Leaderboard",
        "Compare and rank candidates based on matching score and ML prediction"
    )

    if not st.session_state.processed_resumes:
        st.warning("⚠️ No resumes processed yet. Please upload resumes in the **Resume Upload** section first.")
        st.stop()

    # Build leaderboard data
    leaderboard_data = []
    for name, data in st.session_state.processed_resumes.items():
        m = data["metrics"]
        leaderboard_data.append({
            "Candidate Name": name,
            "Match Score": m["match_score"],
            "TF-IDF Similarity (%)": m["tfidf_similarity"],
            "Skill Match (%)": m["skill_overlap_ratio"],
            "Matched Skills": len(m["matched_skills"]),
            "Missing Skills": len(m["missing_skills"]),
            "Experience (Yrs)": m["resume_experience"],
            "Shortlist Prob (%)": round(data["ml_probability"] * 100, 1),
            "Recommendation": "✅ Shortlisted" if data["ml_prediction"] == 1 else "❌ Rejected",
        })

    df_lb = pd.DataFrame(leaderboard_data)
    df_lb = df_lb.sort_values(by="Match Score", ascending=False).reset_index(drop=True)
    df_lb.index = df_lb.index + 1
    df_lb.index.name = "Rank"

    # ── Summary cards ────────────────────────────────────────
    total = len(df_lb)
    shortlisted = len(df_lb[df_lb["Recommendation"] == "✅ Shortlisted"])
    best = df_lb.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("👥", "Total Candidates", str(total), "Resumes screened")
    with c2:
        render_metric_card("✅", "Shortlisted", str(shortlisted), f"of {total} candidates")
    with c3:
        render_metric_card("🏆", "Top Candidate", best["Candidate Name"][:18], f"Score: {best['Match Score']}%")
    with c4:
        avg_score = round(df_lb["Match Score"].mean(), 1)
        render_metric_card("📈", "Avg Match Score", f"{avg_score}%", "Across all candidates")

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

    # ── Leaderboard Table ────────────────────────────────────
    st.markdown("<div class='section-heading'>📋 Candidate Leaderboard</div>", unsafe_allow_html=True)
    st.dataframe(
        df_lb.style.format({
            "Match Score": "{:.1f}",
            "TF-IDF Similarity (%)": "{:.1f}",
            "Skill Match (%)": "{:.1f}",
            "Shortlist Prob (%)": "{:.1f}",
        }),
        use_container_width=True,
        height=min(400, 45 * total + 40),
    )

    # ── Insights ─────────────────────────────────────────────
    st.markdown("<div class='section-heading'>💡 Screening Insights</div>", unsafe_allow_html=True)
    st.markdown(
        f"""<div class='info-panel' style='line-height:1.7; color:#CBD5E1;'>
        🏆 <b>{best['Candidate Name']}</b> leads the board with a match score of <b>{best['Match Score']}%</b>
        and a shortlist probability of <b>{best['Shortlist Prob (%)']:.1f}%</b>.<br>
        📊 Out of <b>{total}</b> screened candidates, <b>{shortlisted}</b>
        {'is' if shortlisted == 1 else 'are'} recommended for shortlisting based on ML prediction criteria.
        </div>""",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — SKILL GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Skill Gap Analysis":
    render_page_header(
        "Skill Gap & Learning Roadmap",
        "Examine missing skills and view targeted learning resources for each candidate"
    )

    if not st.session_state.processed_resumes:
        st.warning("⚠️ No resumes processed yet. Please upload resumes in the **Resume Upload** section first.")
        st.stop()

    selected_candidate = st.selectbox(
        "Select Candidate to Analyze:",
        list(st.session_state.processed_resumes.keys()),
    )

    if selected_candidate:
        data = st.session_state.processed_resumes[selected_candidate]
        m = data["metrics"]
        gap = analyze_skill_gap(m["resume_skills"], st.session_state.jd_skills)

        # ── Candidate overview cards ─────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            color = score_color(m["match_score"])
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='score-ring'>
                        <div class='score-value-large' style='color:{color};'>{m['match_score']}</div>
                        <div class='score-label'>Match Score</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col2:
            render_metric_card("📅", "Experience", f"{m['resume_experience']} yrs", f"Required: {m['jd_experience']} yrs")
        with col3:
            render_metric_card("🎯", "Skill Match", f"{round(m['skill_overlap_ratio'], 1)}%", "Skills overlap ratio")
        with col4:
            prob_pct = round(data["ml_probability"] * 100, 1)
            status = "Likely to Shortlist" if data["ml_prediction"] == 1 else "Unlikely"
            render_metric_card("🤖", "Shortlist Prob", f"{prob_pct}%", status)

        st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

        # ── Skills comparison ────────────────────────────────
        col_match, col_miss = st.columns(2)
        with col_match:
            st.markdown("<div class='section-heading'>✅ Matching Skills</div>", unsafe_allow_html=True)
            if gap["matching"]:
                badges = " ".join(
                    f"<span class='skill-badge skill-badge-match'>✔ {s}</span>"
                    for s in gap["matching"]
                )
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.info("No matching skills found for this candidate.")

        with col_miss:
            st.markdown("<div class='section-heading'>❌ Missing Skills</div>", unsafe_allow_html=True)
            if gap["missing"]:
                badges = " ".join(
                    f"<span class='skill-badge skill-badge-missing'>✘ {s}</span>"
                    for s in gap["missing"]
                )
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.success("🎉 Perfect match — no skills missing!")

        st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

        # ── Learning Roadmap ─────────────────────────────────
        st.markdown("<div class='section-heading'>📚 Recommended Learning Roadmap</div>", unsafe_allow_html=True)

        if gap["recommendations"]:
            st.markdown(
                "<div style='font-size:0.9rem;color:#94A3B8;margin-bottom:1rem;'>"
                "To improve alignment with the job requirements, the candidate should acquire these skills:</div>",
                unsafe_allow_html=True,
            )
            for skill, resource in gap["recommendations"].items():
                st.markdown(
                    f"""<div class='roadmap-item'>
                        <div class='roadmap-skill'>📘 {skill}</div>
                        <div class='roadmap-resource'>{resource}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.success("🌟 Excellent — the candidate possesses all critical skills outlined in the job description!")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Analytics Dashboard":
    render_page_header(
        "Recruitment Analytics & Explainability",
        "Analyze recruitment trends, model performance metrics, and prediction explainability"
    )

    # Load model comparison data (with proper error handling)
    df_comparison = None
    try:
        comp_path = "models/model_comparison.csv"
        if os.path.exists(comp_path):
            df_comparison = pd.read_csv(comp_path, index_col=0)
    except Exception:
        df_comparison = None

    tab1, tab2, tab3 = st.tabs([
        "📊 Candidate Visualization",
        "🧠 Model Performance",
        "🔍 SHAP Explainability",
    ])

    # ── Tab 1: Candidate Visualization ───────────────────────
    with tab1:
        if not st.session_state.processed_resumes:
            st.warning("⚠️ No processed candidates to visualize. Upload resumes first to see analytics.")
        else:
            chart_data = []
            for name, data in st.session_state.processed_resumes.items():
                m_ = data["metrics"]
                chart_data.append({
                    "Name": name,
                    "Match Score": m_["match_score"],
                    "TF-IDF": m_["tfidf_similarity"],
                    "Skill Overlap": m_["skill_overlap_ratio"],
                    "Experience": m_["resume_experience"],
                    "Status": "Shortlisted" if data["ml_prediction"] == 1 else "Rejected",
                })
            df_chart = pd.DataFrame(chart_data)

            col_ch1, col_ch2 = st.columns(2)

            with col_ch1:
                fig_match = px.bar(
                    df_chart.sort_values(by="Match Score", ascending=False),
                    x="Name",
                    y="Match Score",
                    color="Status",
                    color_discrete_map={"Shortlisted": "#10B981", "Rejected": "#F43F5E"},
                    title="Match Score Comparison",
                    labels={"Match Score": "Match Score (%)", "Name": ""},
                )
                fig_match.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#CBD5E1"),
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_match, use_container_width=True)

            with col_ch2:
                fig_scatter = px.scatter(
                    df_chart,
                    x="Experience",
                    y="Match Score",
                    color="Status",
                    size="Skill Overlap",
                    hover_name="Name",
                    title="Experience vs Match Score",
                    labels={"Experience": "Experience (Yrs)", "Match Score": "Score (%)"},
                    color_discrete_map={"Shortlisted": "#10B981", "Rejected": "#F43F5E"},
                )
                fig_scatter.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#CBD5E1"),
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            col_ch3, col_ch4 = st.columns(2)

            with col_ch3:
                all_candidate_skills = []
                for name, data in st.session_state.processed_resumes.items():
                    all_candidate_skills.extend(data["metrics"]["resume_skills"])

                if all_candidate_skills:
                    skill_counts = pd.Series(all_candidate_skills).value_counts().reset_index()
                    skill_counts.columns = ["Skill", "Count"]
                    fig_skills = px.bar(
                        skill_counts,
                        x="Count",
                        y="Skill",
                        orientation="h",
                        title="Skill Frequency Among Candidates",
                        color="Count",
                        color_continuous_scale="Viridis",
                    )
                    fig_skills.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", color="#CBD5E1"),
                        margin=dict(l=20, r=20, t=50, b=20),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_skills, use_container_width=True)
                else:
                    st.info("No skills data available.")

            with col_ch4:
                if st.session_state.jd_skills:
                    missing_skills_all = []
                    for name, data in st.session_state.processed_resumes.items():
                        missing_skills_all.extend(data["metrics"]["missing_skills"])

                    if missing_skills_all:
                        missing_counts = pd.Series(missing_skills_all).value_counts().reset_index()
                        missing_counts.columns = ["Missing Skill", "Count"]
                        fig_missing = px.pie(
                            missing_counts,
                            names="Missing Skill",
                            values="Count",
                            title="Distribution of Missing Required Skills",
                            color_discrete_sequence=px.colors.sequential.RdBu,
                        )
                        fig_missing.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter", color="#CBD5E1"),
                            margin=dict(l=20, r=20, t=50, b=20),
                        )
                        st.plotly_chart(fig_missing, use_container_width=True)
                    else:
                        st.success("🎉 No missing skills — all candidates meet requirements!")
                else:
                    st.info("Configure a Job Description to see missing skills analysis.")

    # ── Tab 2: Model Performance ─────────────────────────────
    with tab2:
        st.markdown("<div class='section-heading'>🧠 Classifier Models Comparison</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='info-panel' style='color:#CBD5E1; font-size:0.92rem; line-height:1.65;'>"
            "Three machine learning classifiers were trained and evaluated on a synthetic dataset of "
            "300 candidate records. The metrics below highlight their evaluation scores on held-out test data."
            "</div>",
            unsafe_allow_html=True,
        )

        if df_comparison is not None and not df_comparison.empty:
            st.markdown("")
            st.dataframe(
                df_comparison.style.format("{:.4f}").highlight_max(axis=0, color="#1a3a2a"),
                use_container_width=True,
            )

            try:
                df_melted = df_comparison.reset_index()
                df_melted = df_melted.rename(columns={df_melted.columns[0]: "Classifier"})
                df_melted = df_melted.melt(id_vars="Classifier", var_name="Metric", value_name="Score")

                fig_compare = px.bar(
                    df_melted,
                    x="Metric",
                    y="Score",
                    color="Classifier",
                    barmode="group",
                    title="Model Performance Metrics Comparison",
                    color_discrete_sequence=["#3B82F6", "#8B5CF6", "#10B981"],
                )
                fig_compare.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#CBD5E1"),
                    margin=dict(l=20, r=20, t=50, b=20),
                    yaxis=dict(range=[0, 1.05]),
                )
                st.plotly_chart(fig_compare, use_container_width=True)
            except Exception as e:
                st.error(f"Could not render comparison chart: {e}")
        else:
            st.warning("Model comparison metrics not found. Please ensure models have been trained by running the data generator.")

    # ── Tab 3: SHAP Explainability ───────────────────────────
    with tab3:
        st.markdown("<div class='section-heading'>🔍 Prediction Explainability (SHAP)</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='info-panel' style='color:#CBD5E1; font-size:0.92rem; line-height:1.65;'>"
            "<b>SHapley Additive exPlanations (SHAP)</b> explains ML predictions by mapping the specific "
            "contribution of each feature — Match Score, Matched Skills Count, Missing Skills Count, and "
            "Experience Years — to the final shortlisting prediction.<br><br>"
            "🔴 <b>Positive SHAP values</b> push the prediction <b>higher</b> (toward shortlisting).<br>"
            "🔵 <b>Negative SHAP values</b> push the prediction <b>lower</b> (toward rejection)."
            "</div>",
            unsafe_allow_html=True,
        )

        if not st.session_state.processed_resumes:
            st.warning("⚠️ Please upload and screen candidates in the **Resume Upload** tab first.")
        elif model is None:
            st.error("❌ ML model not loaded. Predictions and explanations are unavailable.")
        elif X_train_df is None:
            st.error("❌ Training baseline dataset not found. SHAP explanations cannot be generated.")
        else:
            selected_cand_shap = st.selectbox(
                "Select Candidate to Explain Prediction:",
                list(st.session_state.processed_resumes.keys()),
            )

            if selected_cand_shap:
                data = st.session_state.processed_resumes[selected_cand_shap]
                m_ = data["metrics"]

                sample_features = [
                    m_["match_score"],
                    len(m_["matched_skills"]),
                    len(m_["missing_skills"]),
                    m_["resume_experience"],
                ]

                prob_pct = round(data["ml_probability"] * 100, 1)
                status_class = "status-shortlisted" if data["ml_prediction"] == 1 else "status-rejected"
                status_text = "Shortlisted" if data["ml_prediction"] == 1 else "Rejected"
                status_icon = "✅" if data["ml_prediction"] == 1 else "❌"

                st.markdown(
                    f"""<div class='info-panel' style='display:flex;align-items:center;justify-content:space-between;'>
                        <div>
                            <span style='font-size:1.1rem;font-weight:700;color:#F1F5F9;'>
                                {selected_cand_shap}
                            </span>
                            <span style='margin-left:1rem;font-size:0.95rem;color:#94A3B8;'>
                                Shortlist Probability: <b style='color:#F1F5F9;'>{prob_pct}%</b>
                            </span>
                        </div>
                        <span class='{status_class}'>{status_icon} {status_text}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

                with st.spinner("Calculating SHAP values..."):
                    try:
                        fig_shap = get_shap_waterfall_plot(model, X_train_df, sample_features)
                        st.pyplot(fig_shap)
                        plt.close(fig_shap)
                    except Exception as e:
                        st.error(f"Could not generate SHAP visualization: {e}")
