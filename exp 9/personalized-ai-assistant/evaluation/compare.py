from __future__ import annotations

from typing import List


def compare_answers(questions: List[str], base_answers: List[str], rag_answers: List[str], fine_tuned_answers: List[str] | None = None):
    results = []
    for i, question in enumerate(questions):
        modes = [
            {"mode": "Base LLM", "answer": base_answers[i] if i < len(base_answers) else "N/A"},
            {"mode": "RAG", "answer": rag_answers[i] if i < len(rag_answers) else "N/A"},
        ]
        if fine_tuned_answers is not None:
            modes.append({"mode": "Fine-tuned", "answer": fine_tuned_answers[i] if i < len(fine_tuned_answers) else "N/A"})
        results.append({"question": question, "responses": modes})
    return results
