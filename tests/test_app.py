import os
import json
import pickle
import tempfile

import pytest
import numpy as np
import faiss
import requests

import app  


def test_clean_text():
    raw = "Esta é uma linha-\n quebrada   \n\n  com   espaços"
    cleaned = app.clean_text(raw)
    assert "quebrada" in cleaned
    assert "\n" not in cleaned
    assert "  " not in cleaned
    assert cleaned.endswith("espaços")


def test_chunk_text_splits_sentences():
    text = "Frase um. Frase dois! Frase três?"
    chunks = app.chunk_text(text, size=10)

    assert chunks == ["Frase um.", "Frase dois!", "Frase três?"]


def test_build_faiss_index_and_search():
    embs = [[1.0, 0.0], [0.0, 1.0]]
    idx = app.build_faiss_index(embs)
    assert isinstance(idx, faiss.IndexFlatIP)
    assert idx.ntotal == 2

    query = np.array([[1.0, 0.0]], dtype="float32")
    faiss.normalize_L2(query)
    D, I = idx.search(query, 2)
    assert I[0][0] == 0
    assert pytest.approx(D[0][0], rel=1e-3) == 1.0


def test_answer_query_not_ready():
    app.idx = None
    app.status["status"] = "idle"
    with pytest.raises(RuntimeError):
        app.answer_query("teste")


def test_get_semantic_chunks_creates_cache(tmp_path, monkeypatch):
    text = "Um. Dois. Três. Quatro."
    cache_file = tmp_path / "chunks.json"
    monkeypatch.setattr(app, "CHUNK_CACHE_FILE", str(cache_file))

    assert not cache_file.exists()
    chunks = app.get_semantic_chunks(text)
    assert cache_file.exists()

    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    cached_chunks = [item["chunk"] for item in data]
    assert chunks == cached_chunks


def test_extrair_texto_strips_html(monkeypatch):
    html = """
    <html>
      <head><style>body{}</style><script>alert(1)</script></head>
      <body>
        <nav>nav</nav>
        <header>hdr</header>
        <footer>ftr</footer>
        <aside>aside</aside>
        <p>Olá mundo!</p>
        <div> Teste   de    texto </div>
      </body>
    </html>
    """
    class DummyResponse:
        def __init__(self, text):
            self.text = text
        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda url: DummyResponse(html))

    texto = app.extrair_texto("http://fake.url")
    assert "Olá mundo!" in texto
    assert "Teste   de    texto" in texto

    for forbidden in ("alert(1)", "nav", "hdr", "ftr", "aside"):
        assert forbidden not in texto
