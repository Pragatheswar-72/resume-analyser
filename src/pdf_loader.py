"""PDF text extraction.

Takes an uploaded PDF (a file-like object, e.g. Streamlit's UploadedFile)
and returns cleaned plain text. Raises ValueError on unusable input so the
UI can show a friendly message instead of crashing.
"""

import re

import pdfplumber


def _clean(text: str) -> str:
    """Collapse excess whitespace while keeping paragraph breaks."""
    # Normalise line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing spaces on each line.
    text = "\n".join(line.strip() for line in text.split("\n"))
    # Collapse 3+ blank lines into a single blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_text(file) -> str:
    """Extract cleaned text from an uploaded PDF file-like object.

    Args:
        file: A binary file-like object (has .read()) containing a PDF.

    Returns:
        Cleaned plain-text contents of the PDF.

    Raises:
        ValueError: If the file is missing, not a valid PDF, or has no
            extractable text (e.g. a scanned image with no OCR layer).
    """
    if file is None:
        raise ValueError("No file was provided.")

    try:
        # pdfplumber accepts a path or a file-like object.
        with pdfplumber.open(file) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:  # pdfplumber raises various errors on bad input
        raise ValueError(
            "Could not read this file as a PDF. Please upload a valid PDF."
        ) from exc

    text = _clean("\n\n".join(pages))

    if not text:
        raise ValueError(
            "No text could be extracted. The PDF may be a scanned image "
            "without a text layer."
        )

    return text
