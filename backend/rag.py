"""
Lightweight RAG retrieval, no external vector database required.
Uses TF-IDF over the project's own findings documents, plus an optional
PubMed abstract dump if pubmed_abstracts.json is present in this folder.

This is intentionally dependency-light so the whole backend runs on a
free-tier host with no external services. To scale the knowledge base up
later (more literature, more documents), swap this module for a Qdrant-
backed retriever without changing any other part of the backend -- the
retrieve() function signature stays the same.
"""
import glob
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
PUBMED_FILE = os.path.join(os.path.dirname(__file__), "pubmed_abstracts.json")


class RAGIndex:
    def __init__(self):
        self.chunks = []       # list of {"text": str, "source": str, "url": str|None}
        self.vectorizer = None
        self.matrix = None
        self.loaded = False

    def _load_findings(self):
        for path in sorted(glob.glob(os.path.join(KNOWLEDGE_DIR, "*.md"))):
            with open(path, encoding="utf-8") as f:
                text = f.read()
            # chunk by paragraph (blank-line separated)
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
            source_name = os.path.basename(path).replace(".md", "").replace("_", " ")
            for p in paragraphs:
                self.chunks.append({"text": p, "source": f"Project findings: {source_name}", "url": None})

    def _load_pubmed(self):
        if not os.path.exists(PUBMED_FILE):
            return
        with open(PUBMED_FILE, encoding="utf-8") as f:
            articles = json.load(f)
        for a in articles:
            text = f"{a['title']}. {a['abstract']}"
            self.chunks.append({
                "text": text,
                "source": f"PubMed: {a['title']} ({a.get('journal', '')}, {a.get('year', '')})",
                "url": a.get("url"),
            })

    def build(self):
        if self.loaded:
            return
        self._load_findings()
        self._load_pubmed()
        if not self.chunks:
            raise RuntimeError("No knowledge base content found to index.")

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])
        self.loaded = True
        print(f"RAG index built: {len(self.chunks)} chunks "
              f"({'includes PubMed' if os.path.exists(PUBMED_FILE) else 'project findings only'})")

    def retrieve(self, query: str, k: int = 4):
        if not self.loaded:
            self.build()
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]
        results = []
        for i in top_idx:
            if sims[i] <= 0:
                continue
            results.append({
                "text": self.chunks[i]["text"],
                "source": self.chunks[i]["source"],
                "url": self.chunks[i]["url"],
                "score": float(sims[i]),
            })
        return results


rag_index = RAGIndex()
