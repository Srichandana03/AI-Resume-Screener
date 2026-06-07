"""
PDF Parser Module
=================
Extracts raw text content from PDF files using the pypdf library.

This module handles both file paths and file-like objects (e.g., Streamlit
UploadedFile or BytesIO), making it compatible with local files and web uploads.

Functions:
    extract_text_from_pdf(pdf_file) -> str
"""

import pypdf


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract raw text from a PDF file.

    Iterates through all pages of the PDF, extracts text content,
    and performs basic line-level cleanup (strips whitespace, removes
    empty lines).

    Parameters
    ----------
    pdf_file : str or file-like object
        Path to the PDF file on disk, or a file-like object such as
        a Streamlit UploadedFile or io.BytesIO instance.

    Returns
    -------
    str
        Cleaned extracted text with one non-empty line per newline.
        Returns an empty string if the PDF cannot be read.
    """
    text = ""
    try:
        reader = pypdf.PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        # Gracefully handle corrupted/unreadable PDFs
        print(f"[pdf_parser] Error reading PDF: {e}")

    # Strip trailing/leading whitespace and filter out empty lines
    cleaned_lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(cleaned_lines)
