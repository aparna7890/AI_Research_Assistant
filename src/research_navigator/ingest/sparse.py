"""BM25-style sparse vectors for hybrid search."""
from __future__ import annotations
import math
import re
from collections import Counter

_STOPWORDS = frozenset(
    "a an the and or but in on at to for of with is are was were be been "
    "have has had do does did will would could should may might can".split()
)


def tokenise(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def build_sparse_vector(text: str) -> dict[str, list[int] | list[float]]:
    tokens = tokenise(text)
    if not tokens:
        return {"indices": [], "values": []}
    tf = Counter(tokens)
    doc_len = len(tokens)
    
    seen: dict[int, float] = {}
    for token, count in tf.items():
        idx = _hash_token(token)
        score = round((count / doc_len) * (1 + math.log(1 + count)), 6)
        # If two tokens hash to the same index, add their scores together
        seen[idx] = seen.get(idx, 0.0) + score

    return {
        "indices": list(seen.keys()),
        "values": list(seen.values()),
    }


def _hash_token(token: str) -> int:
    h = 2166136261
    for ch in token.encode():
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h % (2**20)
