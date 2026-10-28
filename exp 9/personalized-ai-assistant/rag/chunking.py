from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text.strip():
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == len(text):
            break
        start += chunk_size - overlap

    return [chunk for chunk in chunks if chunk]


def build_chunks_from_documents(documents: List[dict], chunk_size: int, overlap: int) -> List[dict]:
    chunks = []
    for document in documents:
        filename = document.get("filename", "unknown")
        text = document.get("text", "")
        pages = document.get("pages", [])
        page_map = {page_num: page_text for page_num, page_text in pages}
        chunk_list = chunk_text(text, chunk_size, overlap)
        for index, chunk in enumerate(chunk_list):
            page_number = 1
            for page_num, page_text in pages:
                if chunk in page_text:
                    page_number = page_num
                    break
            chunks.append({
                "filename": filename,
                "chunk_index": index,
                "text": chunk,
                "source": filename,
                "page_number": page_number,
            })
    return chunks
