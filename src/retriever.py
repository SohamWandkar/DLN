"""
Retriever Engine for LearnMate
Implements Hybrid Retrieval fusing Dense Semantic Embeddings and BM25 Lexical Matching.
"""
import os
import sys
import math
import re
from typing import List, Dict, Any, Tuple
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    EMBEDDING_MODEL_NAME, TOP_K_RETRIEVAL, RRF_K, 
    WEIGHT_DENSE, WEIGHT_BM25, RANDOM_SEED
)

class BM25Retriever:
    """
    Pure Python BM25Okapi Implementation for full portability and deterministic reproducibility.
    """
    def __init__(self, corpus: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.avg_doc_len = 0.0
        self.doc_freqs = []
        self.idf = {}
        self.corpus_size = len(corpus)
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase, alphanumeric tokenization
        tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
        return tokens

    def _build_index(self):
        df = {}
        total_len = 0
        for doc in self.corpus:
            tokens = self._tokenize(doc["text"])
            self.doc_len.append(len(tokens))
            total_len += len(tokens)
            freqs = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)
            for t in freqs:
                df[t] = df.get(t, 0) + 1

        self.avg_doc_len = total_len / max(1, self.corpus_size)
        for t, freq in df.items():
            # Standard Robertson-Spärck Jones IDF
            self.idf[t] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query: str) -> List[Tuple[int, float]]:
        tokens = self._tokenize(query)
        scores = []
        for i, freqs in enumerate(self.doc_freqs):
            score = 0.0
            dl = self.doc_len[i]
            for t in tokens:
                if t in freqs:
                    tf = freqs[t]
                    idf = self.idf.get(t, 0.0)
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(1e-6, self.avg_doc_len)))
                    score += idf * (numerator / denominator)
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class DenseRetriever:
    """
    Dense Bi-Encoder Retriever using Sentence Transformers.
    """
    def __init__(self, corpus: List[Dict[str, Any]], model_name: str = EMBEDDING_MODEL_NAME):
        self.corpus = corpus
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading dense embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            texts = [f"{d['module']} {d['section']} {d['text']}" for d in self.corpus]
            self.embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            print(f"Computed dense embeddings for {len(self.corpus)} chunks. Shape: {self.embeddings.shape}")
        except Exception as e:
            print(f"SentenceTransformer initialization notice: {e}. Utilizing high-dimensional semantic TF-IDF dense projection fallback.")
            self._init_fallback_embeddings()

    def _init_fallback_embeddings(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(max_features=384, stop_words="english", ngram_range=(1, 2))
        texts = [f"{d['module']} {d['section']} {d['text']}" for d in self.corpus]
        raw_mat = self.tfidf.fit_transform(texts).toarray()
        norms = np.linalg.norm(raw_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = raw_mat / norms

    def score(self, query: str) -> List[Tuple[int, float]]:
        if self.model is not None:
            q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        else:
            q_mat = self.tfidf.transform([query]).toarray()
            q_norm = np.linalg.norm(q_mat)
            q_emb = q_mat[0] / max(1e-6, q_norm)

        # Cosine similarity
        sims = np.dot(self.embeddings, q_emb)
        ranked = [(i, float(sims[i])) for i in range(len(self.corpus))]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


class HybridRetriever:
    """
    Hybrid Retriever fusing BM25 and Dense Bi-Encoder via Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, corpus: List[Dict[str, Any]]):
        self.corpus = corpus
        self.bm25 = BM25Retriever(corpus)
        self.dense = DenseRetriever(corpus)

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        bm25_scores = self.bm25.score(query)
        dense_scores = self.dense.score(query)

        # Build rank maps
        bm25_ranks = {doc_idx: rank for rank, (doc_idx, _) in enumerate(bm25_scores)}
        dense_ranks = {doc_idx: rank for rank, (doc_idx, _) in enumerate(dense_scores)}
        bm25_score_map = {doc_idx: score for doc_idx, score in bm25_scores}
        dense_score_map = {doc_idx: score for doc_idx, score in dense_scores}

        # Reciprocal Rank Fusion
        fused = []
        for i in range(len(self.corpus)):
            r_dense = dense_ranks[i]
            r_bm25 = bm25_ranks[i]
            rrf_score = (WEIGHT_DENSE / (RRF_K + r_dense + 1)) + (WEIGHT_BM25 / (RRF_K + r_bm25 + 1))
            fused.append((i, rrf_score))

        fused.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_idx, rrf in fused[:top_k]:
            item = dict(self.corpus[doc_idx])
            item["rrf_score"] = float(rrf)
            item["dense_score"] = float(dense_score_map[doc_idx])
            item["bm25_score"] = float(bm25_score_map[doc_idx])
            results.append(item)

        return results

if __name__ == "__main__":
    import json
    from src.config import PROCESSED_DIR
    chunks_file = os.path.join(PROCESSED_DIR, "chunks_corpus.json")
    with open(chunks_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)["chunks"]

    retriever = HybridRetriever(corpus)
    test_q = "How does Nesterov accelerated gradient prevent oscillations?"
    hits = retriever.retrieve(test_q, top_k=2)
    print(f"\nQuery: {test_q}")
    for h in hits:
        print(f"-> [{h['citation']}] (Dense: {h['dense_score']:.3f}, BM25: {h['bm25_score']:.2f}, RRF: {h['rrf_score']:.4f})")
        print(f"   Excerpt: {h['text'][:120]}...\n")
