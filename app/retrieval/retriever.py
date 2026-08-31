"""
Search the knowledge base for relevant documents.

The scoring works like this:
  Score = How similar the document is × How trustworthy the document is

Old, internal, or draft documents are filtered out before results are shown.
"""

from __future__ import annotations

from typing import Any
import sys, os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TOP_K_CHUNKS, SIMILARITY_THRESHOLD
from retrieval.ingest import load_all_chunks
from retrieval.authority import is_eligible, authority_score, detect_conflict


class AuthorityAwareRetriever:
    """
    Index knowledge base documents and search them.
    
    How it works:
      1. Measure how similar each document is to the question
      2. Multiply that by a trust score (old docs get lower trust)
      3. Only keep results above a minimum quality level
      4. Return the top results
    """

    def __init__(self) -> None:
        self._chunks: list[dict[str, Any]] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None  # sparse matrix
        self._eligible_mask: list[bool] = []

    def build_index(self) -> None:
        """Read all knowledge documents and prepare them for searching."""
        self._chunks = load_all_chunks()
        texts = [c["text"] for c in self._chunks]
        self._eligible_mask = [is_eligible(c) for c in self._chunks]

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def retrieve(
        self, query: str, top_k: int = TOP_K_CHUNKS
    ) -> dict[str, Any]:
        """
        Return a dict with:
          - 'chunks': list of top-K eligible chunks (with 'score' key added)
          - 'conflict': bool
          - 'conflict_description': str
          - 'excluded_count': int (how many chunks were filtered out)
        """
        if self._vectorizer is None:
            self.build_index()

        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix).flatten()

        scored: list[dict[str, Any]] = []
        excluded_count = 0

        for idx, (chunk, eligible) in enumerate(zip(self._chunks, self._eligible_mask)):
            raw_sim = float(sims[idx])
            if not eligible:
                excluded_count += 1
                continue  # hard-exclude superseded / internal / draft
            auth = authority_score(chunk)
            final_score = raw_sim * auth
            if final_score < SIMILARITY_THRESHOLD:
                continue
            scored.append({**chunk, "score": round(final_score, 4),
                           "raw_sim": round(raw_sim, 4)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:top_k]

        conflict, conflict_desc = detect_conflict(top)

        return {
            "chunks": top,
            "conflict": conflict,
            "conflict_description": conflict_desc,
            "excluded_count": excluded_count,
        }

    def get_all_eligible_chunks(self) -> list[dict[str, Any]]:
        """Return all eligible chunks (for debugging / inspection)."""
        if not self._chunks:
            self.build_index()
        return [c for c, e in zip(self._chunks, self._eligible_mask) if e]


# Singleton instance — built lazily
_retriever: AuthorityAwareRetriever | None = None


def get_retriever() -> AuthorityAwareRetriever:
    global _retriever
    if _retriever is None:
        _retriever = AuthorityAwareRetriever()
        _retriever.build_index()
    return _retriever


if __name__ == "__main__":
    r = get_retriever()
    result = r.retrieve("how long do I have to return my backpack")
    print(f"Retrieved {len(result['chunks'])} chunks, {result['excluded_count']} excluded")
    for c in result["chunks"]:
        print(f"  [{c['score']:.3f}] {c['source_ref']}")
    if result["conflict"]:
        print(f"\nCONFLICT: {result['conflict_description']}")
