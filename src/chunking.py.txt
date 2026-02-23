from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping word-based chunks.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def create_chunks(pages: List[Dict], chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """
    Convert page-level parsed data into chunk-level structured data.
    """
    all_chunks = []
    chunk_id = 0

    for page in pages:
        text_chunks = chunk_text(page["text"], chunk_size, overlap)

        for chunk in text_chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "document_name": page["document_name"],
                "page_number": page["page_number"],
                "chunk_text": chunk
            })
            chunk_id += 1

    return all_chunks