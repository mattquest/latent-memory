"""Deterministic lexical retrieval; never reads gold answers or supporting IDs."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re

STOPWORDS = frozenset("a an the and or of in on at to for from by with is are was were be been being do does did have has had it its this that these those i me my we our you your they their he she as can could would should what which who whom where when how please tell about more before after".split())


def terms(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9]+", text.lower())
            if term not in STOPWORDS and len(term) > 1]


class BM25Index:
    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)
        self.lengths = []
        for i, document in enumerate(documents):
            counts = Counter(terms(document.get("title", "") + "\n" + document["text"]))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((i, frequency))
        self.average_length = sum(self.lengths) / max(1, len(documents)) or 1

    def search(self, question: str, k: int = 20, max_per_source: int = 2) -> list[tuple[dict, float]]:
        if k < 1 or max_per_source < 1:
            raise ValueError("retrieval caps must be positive")
        scores = defaultdict(float)
        for term in set(terms(question)):
            posting = self.postings.get(term, ())
            idf = math.log(1 + (len(self.documents) - len(posting) + 0.5) / (len(posting) + 0.5))
            for index, frequency in posting:
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * self.lengths[index] / self.average_length)
                scores[index] += idf * frequency * (self.k1 + 1) / denominator
        # Stable source order resolves ties; IDs may contain benchmark support
        # annotations and must never influence ranking. Zero-score documents
        # remain available when an experiment expands its evidence budget.
        ranking = sorted(range(len(self.documents)), key=lambda index: (-scores[index], index))
        result, source_counts = [], Counter()
        for index in ranking:
            document = self.documents[index]
            source = document.get("source_message_id", document["id"])
            if source_counts[source] >= max_per_source:
                continue
            result.append((document, scores[index]))
            source_counts[source] += 1
            if len(result) >= k:
                break
        return result
