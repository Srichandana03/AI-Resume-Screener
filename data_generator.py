"""
Synthetic Data & Mock Resume Generator
======================================
Generates synthetic dataset of candidate features for training classification models,
and creates sample recruiter assets including PDF resumes and Job Descriptions.

Functions:
    generate_csv_dataset(filepath, n_samples)
    create_pdf_resume(filename, name, title, contact, experience, skills, education)
    generate_mock_data()
"""

import os
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from utils.ml_model import train_and_evaluate

def generate_csv_dataset(filepath: str, n_samples: int = 300):
    """
    Generate a synthetic dataset with realistic feature correlations.

    Creates features (similarity_score, matched_skills_count,
    missing_skills_count, experience_years) and uses a logistic-like sigmoid
    function with 7% added noise to generate the 'shortlisted' label.

    Parameters
    ----------
    filepath : str
        Target CSV file path to save the dataset.
    n_samples : int, optional
        Number of candidate rows to generate (default is 300).
    """
    np.random.seed(42)
    
    # Features
    similarity_score = np.random.normal(loc=62, scale=12, size=n_samples)
    similarity_score = np.clip(similarity_score, 15, 100)
    
    # Matched skills count (usually correlated with similarity)
    matched_skills_count = np.round(
        0.1 * similarity_score + np.random.poisson(lam=1.5, size=n_samples)
    ).astype(int)
    matched_skills_count = np.clip(matched_skills_count, 0, 12)
    
    # Missing skills count
    missing_skills_count = np.random.poisson(lam=4, size=n_samples)
    missing_skills_count = np.clip(missing_skills_count, 0, 12)
    
    # Experience years
    experience_years = np.random.exponential(scale=3.5, size=n_samples).astype(int)
    experience_years = np.clip(experience_years, 0, 15)
    
    # Calculate probability score for shortlisting
    # Weighted linear combination
    score = (
        0.05 * (similarity_score - 60) +
        0.3 * (matched_skills_count - 3) -
        0.15 * (missing_skills_count - 4) +
        0.2 * (experience_years - 2)
    )
    
    # Sigmoid function for probability
    prob = 1 / (1 + np.exp(-score))
    
    # Final labels
    shortlisted = (prob > 0.5).astype(int)
    
    # Add noise (flip ~7% of records)
    noise = np.random.rand(n_samples) < 0.07
    shortlisted[noise] = 1 - shortlisted[noise]
    
    # Create DataFrame
    df = pd.DataFrame({
        "similarity_score": np.round(similarity_score, 1),
        "matched_skills_count": matched_skills_count,
        "missing_skills_count": missing_skills_count,
        "experience_years": experience_years,
        "shortlisted": shortlisted
    })
    
    df.to_csv(filepath, index=False)
    print(f"Generated synthetic training dataset: {filepath} ({n_samples} rows)")

def create_pdf_resume(
    filename: str,
    name: str,
    title: str,
    contact: str,
    experience: list,
    skills: list,
    education: str
):
    """
    Create a formal PDF resume using the ReportLab layout engine.

    Generates a structured multi-section document containing the candidate's
    name, professional title, contact info, professional experience timeline,
    technical skills list, and education history.

    Parameters
    ----------
    filename : str
        Target PDF file path (e.g., "resumes/candidate.pdf").
    name : str
        Candidate's full name.
    title : str
        Professional role/title (e.g., "Senior ML Engineer").
    contact : str
        Contact details string containing email, phone, location, etc.
    experience : list[str]
        List of professional experience descriptions formatted as HTML strings.
    skills : list[str]
        List of technical skills to display.
    education : str
        Education credentials string.
    """
    doc = SimpleDocTemplate(
        filename, 
        pagesize=letter, 
        rightMargin=45, 
        leftMargin=45, 
        topMargin=45, 
        bottomMargin=45
    )
    styles = getSampleStyleSheet()
    
    # Custom styling
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontSize=22, leading=26, textColor='#1E3A8A'
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Heading2'], fontSize=13, leading=17, textColor='#4B5563'
    )
    section_style = ParagraphStyle(
        'SectionHeading', parent=styles['Heading3'], fontSize=11, leading=15, textColor='#1E3A8A', spaceBefore=8, spaceAfter=4
    )
    body_style = ParagraphStyle(
        'BodyTextCustom', parent=styles['Normal'], fontSize=9.5, leading=13.5, spaceAfter=4
    )
    
    story = []
    
    # Header
    story.append(Paragraph(f"<b>{name}</b>", title_style))
    story.append(Paragraph(title, subtitle_style))
    story.append(Paragraph(contact, body_style))
    story.append(Spacer(1, 8))
    
    # Experience
    story.append(Paragraph("<b>PROFESSIONAL EXPERIENCE</b>", section_style))
    for exp in experience:
        story.append(Paragraph(exp, body_style))
    story.append(Spacer(1, 8))
    
    # Skills
    story.append(Paragraph("<b>TECHNICAL SKILLS</b>", section_style))
    skills_text = ", ".join(skills)
    story.append(Paragraph(skills_text, body_style))
    story.append(Spacer(1, 8))
    
    # Education
    story.append(Paragraph("<b>EDUCATION</b>", section_style))
    story.append(Paragraph(education, body_style))
    
    doc.build(story)
    print(f"Created PDF resume: {filename}")

def generate_mock_data():
    """
    Generate all mock assets including candidate PDF resumes and job descriptions.

    Ensures the target directories `data/` and `resumes/` exist, calls
    `generate_csv_dataset` to build the classification training CSV, creates
    6 mock candidate PDF resumes with diverse skills/experience profiles, and
    saves 2 baseline job descriptions (ML Engineer and Full-Stack React).
    """
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("resumes", exist_ok=True)
    
    # 1. Generate dataset
    generate_csv_dataset("data/candidates.csv")
    
    # 2. Sample Resumes Data
    resumes = [
        {
            "filename": "resumes/alex_carter_ml_engineer.pdf",
            "name": "Alex Carter",
            "title": "Machine Learning Engineer",
            "contact": "Email: alex.carter@email.com | Phone: +1-555-0199 | Location: San Francisco, CA",
            "experience": [
                "<b>Senior ML Engineer at Alpha AI Corp (2023 - Present)</b><br/>"
                "Designed and implemented production-grade Machine Learning pipelines. "
                "Utilized PyTorch and TensorFlow for Deep Learning image classification. "
                "Worked extensively in Python for model scripting. 3 years of experience in leading projects.",
                "<b>Data Scientist at Beta Analytics (2021 - 2023)</b><br/>"
                "Built predictive models using Scikit-Learn. "
                "Developed complex data pipelines utilizing SQL to extract insights from warehouses. "
                "2 years of experience analyzing large-scale datasets."
            ],
            "skills": ["Python", "Machine Learning", "Deep Learning", "SQL", "TensorFlow", "PyTorch", "Scikit-Learn", "Git", "Data Structures", "Algorithms"],
            "education": "Master of Science in Computer Science, Stanford University (2021)"
        },
        {
            "filename": "resumes/sarah_jenkins_frontend.pdf",
            "name": "Sarah Jenkins",
            "title": "Senior Frontend Developer",
            "contact": "Email: sarah.j@email.com | Phone: +1-555-0144 | Location: Austin, TX",
            "experience": [
                "<b>Lead Frontend Developer at WebFlow Solutions (2022 - Present)</b><br/>"
                "Engineered responsive user interfaces using React and modern JavaScript. "
                "Implemented interactive design systems using CSS and HTML. "
                "Managed build releases and branch structuring using Git. 4 years of experience.",
                "<b>React Developer at Interface Studio (2020 - 2022)</b><br/>"
                "Developed modular component UI libraries. "
                "Optimized app performance for web clients using React Hooks. "
                "2 years of experience in frontend systems."
            ],
            "skills": ["React", "JavaScript", "Git", "Data Structures", "Algorithms"],
            "education": "Bachelor of Science in Software Engineering, UT Austin (2020)"
        },
        {
            "filename": "resumes/michael_chang_devops.pdf",
            "name": "Michael Chang",
            "title": "DevOps & Cloud Engineer",
            "contact": "Email: mchang@email.com | Phone: +1-555-0166 | Location: Seattle, WA",
            "experience": [
                "<b>Cloud Platform Engineer at CloudScale Systems (2021 - Present)</b><br/>"
                "Managed cloud infrastructure on AWS. "
                "Containerized complex microservices utilizing Docker and configured CI/CD. "
                "Wrote automation scripts using Python and Git. 5 years of experience.",
                "<b>System Administrator at InfraCorp (2018 - 2021)</b><br/>"
                "Maintained local servers and network directories. "
                "Automated regular system backups. 3 years of experience."
            ],
            "skills": ["Docker", "AWS", "Git", "Python", "SQL", "FastAPI"],
            "education": "Bachelor of Science in Computer Engineering, University of Washington (2018)"
        },
        {
            "filename": "resumes/emily_watson_backend.pdf",
            "name": "Emily Watson",
            "title": "Backend Software Developer",
            "contact": "Email: emily.watson@email.com | Phone: +1-555-0188 | Location: Chicago, IL",
            "experience": [
                "<b>Software Engineer II at FinTech Systems (2022 - Present)</b><br/>"
                "Built enterprise backend services using Java. "
                "Designed relational database schemas with SQL and optimized indexing. "
                "Created API services utilizing Java frameworks and FastAPI. 3 years of experience.",
                "<b>Junior Developer at CodeBase LLC (2020 - 2022)</b><br/>"
                "Assisted in backend codebase maintenance. "
                "Implemented core Data Structures and Algorithms for search optimization. "
                "2 years of experience."
            ],
            "skills": ["Java", "SQL", "FastAPI", "Git", "Data Structures", "Algorithms", "Python"],
            "education": "Bachelor of Science in Computer Science, University of Illinois (2020)"
        },
        {
            "filename": "resumes/john_doe_intern.pdf",
            "name": "John Doe",
            "title": "Junior Python Developer",
            "contact": "Email: john.doe@email.com | Phone: +1-555-0122 | Location: Boston, MA",
            "experience": [
                "<b>Software Intern at TechStart (Summer 2025)</b><br/>"
                "Wrote basic script utilities in Python. "
                "Worked with team members using Git for version control. "
                "Assisted in querying standard relational databases using SQL."
            ],
            "skills": ["Python", "SQL", "Git"],
            "education": "Undergraduate Student in CS, Boston University (Expected 2027)"
        },
        {
            "filename": "resumes/david_miller_ml_junior.pdf",
            "name": "David Miller",
            "title": "Data Scientist & ML Developer",
            "contact": "Email: dmiller@email.com | Phone: +1-555-0111 | Location: Denver, CO",
            "experience": [
                "<b>Data Science Analyst at Apex Data (2024 - Present)</b><br/>"
                "Constructed predictive classifiers with Python and Scikit-Learn. "
                "Explored text analysis tasks using NLP libraries. "
                "Wrote queries with SQL. 2 years of experience."
            ],
            "skills": ["Python", "Machine Learning", "NLP", "SQL", "Scikit-Learn", "Git"],
            "education": "Bachelor of Science in Data Science, Colorado State University (2024)"
        }
    ]
    
    for r in resumes:
        create_pdf_resume(
            r["filename"], r["name"], r["title"], r["contact"], 
            r["experience"], r["skills"], r["education"]
        )
        
    # 3. Create Sample Job Descriptions
    jd_ml = (
        "Role: Machine Learning Engineer\n"
        "Requirements:\n"
        "- Strong programming skills in Python.\n"
        "- Experience in Machine Learning and Deep Learning frameworks like PyTorch or TensorFlow.\n"
        "- Experience training models with Scikit-Learn.\n"
        "- Strong understanding of Data Structures and Algorithms.\n"
        "- SQL database query experience.\n"
        "- Version control using Git.\n"
        "- Minimum 3+ years of experience in similar roles."
    )
    with open("data/jd_ml_engineer.txt", "w") as f:
        f.write(jd_ml)
        
    jd_web = (
        "Role: Full-Stack React & Node Developer\n"
        "Requirements:\n"
        "- Highly proficient in frontend frameworks like React and JavaScript.\n"
        "- Understanding of Data Structures and Algorithms.\n"
        "- Version control using Git.\n"
        "- Python and FastAPI backend development is a strong plus.\n"
        "- Cloud experience (AWS) and containerization (Docker) are beneficial.\n"
        "- 2+ years of experience."
    )
    with open("data/jd_web_developer.txt", "w") as f:
        f.write(jd_web)
        
    print("Mock PDF resumes and job description files generated successfully!")

if __name__ == "__main__":
    # Change working directory if run from root directory
    generate_mock_data()
    print("Training ML Models on generated dataset...")
    # Train ML models immediately using our new utility
    train_and_evaluate("data/candidates.csv", "models")
    print("Models trained and saved successfully in models/ directory!")
