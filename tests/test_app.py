import os
import json
import pickle

import pytest
import numpy as np
import faiss
import requests
from fpdf import FPDF

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

    q = np.array([[1.0, 0.0]], dtype="float32")
    faiss.normalize_L2(q)
    D, I = idx.search(q, 2)
    assert I[0][0] == 0
    assert D[0][0] == pytest.approx(1.0, rel=1e-3)


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
    assert chunks == [item["chunk"] for item in data]


def test_load_pdf_clean(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Linha1-\nLinha2")
    pdf.output(str(pdf_path))

    text = app.load_pdf_clean(str(pdf_path))
    assert "Linha1Linha2" in text


def test_get_embeddings_creates_cache(tmp_path, monkeypatch):
    chunks = ["texto1", "texto2"]
    cache_file = tmp_path / "embeddings.pkl"
    monkeypatch.setattr(app, "EMBED_CACHE_FILE", str(cache_file))

    class DummyItem:
        def __init__(self, embedding):
            self.embedding = embedding

    class DummyResp:
        def __init__(self):
            self.data = [DummyItem([0.5, 0.5]), DummyItem([1.0, 0.0])]

    class DummyEmbeddings:
        @staticmethod
        def create(model, input):
            return DummyResp()

    class DummyClient:
        embeddings = DummyEmbeddings()

    monkeypatch.setattr(app, "client", DummyClient())

    embs = app.get_embeddings(chunks)
    assert cache_file.exists()

    with open(cache_file, "rb") as f:
        stored = pickle.load(f)
    assert set(stored.keys()) == set(chunks)
    assert embs == [stored[c] for c in chunks]


def test_answer_query_ready(monkeypatch):
    app.status["status"] = "ready"

    monkeypatch.setattr(app, "get_semantic_chunks", lambda q: ["ctx"])
    monkeypatch.setattr(app, "get_embeddings", lambda cs: [[1.0, 0.0]])

    class DummyQItem:
        def __init__(self, embedding):
            self.embedding = embedding

    class DummyQResp:
        def __init__(self):
            self.data = [DummyQItem([1.0, 0.0])]

    class DummyEmbModel:
        @staticmethod
        def create(model, input):
            return DummyQResp()

    class DummyChoice:
        def __init__(self):
            self.message = type("m", (), {"content": "IA respondeu"})

    class DummyChatComp:
        @staticmethod
        def create(model, messages, temperature):
            resp = type("r", (), {})()
            resp.choices = [DummyChoice()]
            return resp

    class DummyChat:
        completions = DummyChatComp()

    class DummyClient:
        embeddings = DummyEmbModel()
        chat = DummyChat()

    monkeypatch.setattr(app, "client", DummyClient())

    class DummyIndex:
        def search(self, array, k):
            return None, np.array([[0]])

    app.idx = DummyIndex()

    resp = app.answer_query("pergunta", k=1)
    assert resp == "IA respondeu"


def test_extrair_texto_strips_html(monkeypatch):
    html = """
    <html>
      <head><style>body{}</style><script>alert(1)</script></head>
      <body>
        <nav>nav</nav><header>hdr</header><footer>ftr</footer><aside>aside</aside>
        <p>Olá mundo!</p><div> Teste   de    texto </div>
      </body>
    </html>
    """
    class DummyResponse:
        def __init__(self, text): self.text = text
        def raise_for_status(self): pass

    monkeypatch.setattr(requests, "get", lambda url: DummyResponse(html))
    texto = app.extrair_texto("https://fake")
    assert "Olá mundo!" in texto
    assert "Teste   de    texto" in texto
    for forbidden in ("alert(1)", "nav", "hdr", "ftr", "aside"):
        assert forbidden not in texto


def test_gerar_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "extrair_texto", lambda url: "conteúdo")
    output = tmp_path / "out.pdf"
    app.gerar_pdf(["url1"], str(output))
    assert output.exists() and output.stat().st_size > 0


def test_process_urls_transitions(monkeypatch):
    monkeypatch.setattr(app, "gerar_pdf", lambda urls, out: None)
    monkeypatch.setattr(app, "load_pdf_clean", lambda path: "txt")
    monkeypatch.setattr(app, "get_semantic_chunks", lambda t: ["c"])
    monkeypatch.setattr(app, "get_embeddings", lambda c: [[0.1]])
    monkeypatch.setattr(app, "build_faiss_index", lambda e: faiss.IndexFlatIP(1))
    monkeypatch.setattr(app, "answer_query", lambda q, k: "resp")

    app.status["status"] = "idle"
    app.process_urls(["u1", "u2"])
    assert app.status["status"] == "ready"


class TestFlaskEndpoints:
    @pytest.fixture(autouse=True)
    def client(self):
        return app.app.test_client()

    def test_process_endpoint(self, client):
        resp = client.post("/process", json={"urls": ["http://x"]})
        assert resp.status_code == 202
        data = resp.get_json()
        assert data["message"] == "Processamento iniciado"

    def test_status_endpoint(self, client):
        app.status["status"] = "processing"
        resp = client.get("/status")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "processing"

    def test_query_endpoint_not_ready(self, client):
        app.status["status"] = "idle"
        resp = client.post("/query", json={"query": "q"})
        assert resp.status_code == 400

    def test_query_endpoint_ready(self, client, monkeypatch):
        app.status["status"] = "ready"
        monkeypatch.setattr(app, "answer_query", lambda q, k: "ok")
        resp = client.post("/query", json={"query": "q"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 1
        assert list(data.values())[0] == "ok"