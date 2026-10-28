from __future__ import annotations

import faiss
import numpy as np


class FAISSVectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def add(self, embeddings, metadata):
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata length mismatch")
        if len(embeddings) == 0:
            return
        np_embeddings = np.asarray(embeddings, dtype="float32")
        self.index.add(np_embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embedding, top_k: int):
        query = np.asarray([query_embedding], dtype="float32")
        distances, indices = self.index.search(query, min(top_k, len(self.metadata)))
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0:
                results.append({"index": int(idx), "distance": float(distance), "metadata": self.metadata[int(idx)]})
        return results

    def reset(self):
        self.index = faiss.IndexFlatL2(self.index.d)
        self.metadata = []
