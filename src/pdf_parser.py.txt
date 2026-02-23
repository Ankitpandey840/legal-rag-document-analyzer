import os
import pdfplumber
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract page-wise text from a single PDF.
    """
    pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages_data.append({
                    "document_name": os.path.basename(pdf_path),
                    "page_number": page_num,
                    "text": text.strip()
                })

    return pages_data


def parse_pdf_directory(root_dir: str) -> List[Dict]:
    """
    Recursively parse all PDFs inside a directory.
    """
    all_pages = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                pages = extract_text_from_pdf(pdf_path)
                all_pages.extend(pages)

    return all_pages