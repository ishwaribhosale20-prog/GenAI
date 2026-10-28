from __future__ import annotations

from pathlib import Path
from typing import List

from docx import Document
from pypdf import PdfReader


class DocumentLoadError(ValueError):
    """Raised when a file cannot be processed."""


def read_text_file(path: str | Path) -> dict:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        text = file_path.read_text(encoding="utf-8", errors="replace")
        return {"filename": file_path.name, "text": text, "pages": [(1, text)]}

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append((page_number, text))
        combined_text = "\n\n".join(text for _, text in pages)
        return {"filename": file_path.name, "text": combined_text, "pages": pages}

    if suffix == ".docx":
        doc = Document(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        return {"filename": file_path.name, "text": text, "pages": [(1, text)]}

    raise DocumentLoadError(f"Unsupported file format: {file_path.suffix}")


def extract_text_from_files(file_paths: List[str | Path]) -> List[dict]:
    records: List[dict] = []
    for file_path in file_paths:
        resolved = Path(file_path)
        try:
            document = read_text_file(resolved)
        except Exception as exc:  # pragma: no cover - defensive validation
            records.append({"filename": resolved.name, "text": "", "pages": [], "error": str(exc)})
            continue

        cleaned = document["text"].strip()
        if not cleaned:
            records.append({"filename": resolved.name, "text": "", "pages": [], "error": "Document is empty or unreadable."})
            continue

        document["error"] = None
        records.append(document)
    return records
