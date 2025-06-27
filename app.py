import os
import re
import json
import pickle
import logging
import threading
import requests

from typing import List
from dotenv import load_dotenv
from flask import Flask, request
from flask_restx import Api, Resource, fields
from bs4 import BeautifulSoup
from fpdf import FPDF
import pdfplumber
import faiss
import numpy as np
from openai import OpenAI

# --- Configuração de logs ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Carrega variáveis de ambiente ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Paths e modelos ---
CHUNK_MODEL      = "gpt-4"
EMBED_MODEL      = "text-embedding-3-large"
CHAT_MODEL       = "gpt-3.5-turbo"
DEFAULT_PDF_PATH = "./documentacao.pdf"
CHUNK_CACHE_FILE = "semantic_chunks.json"
EMBED_CACHE_FILE = "embeddings_cache.pkl"
CHAT_CACHE_FILE  = "chat_cache.pkl"
FONTS_DIR        = os.path.join(os.path.dirname(__file__), "fonts")

# --- Estado global de processing ---
status = {
    "status": "idle",    # idle, queued, generating_pdf, reading_pdf, chunking, embedding, indexing, ready, error
    "progress": 0,
    "message": None
}

# --- Contexto RAG (inicialmente vazio) ---
embed_cache: dict = {}
chat_cache: dict  = {}
idx = None
blocos: List[str] = []

# --- Utilitários de texto/PDF/embedding ---

def clean_text(text: str) -> str:
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s*\n\s*', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_pdf_clean(path: str) -> str:
    logger.info("Loading & cleaning PDF: %s", path)
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            raw = p.extract_text() or ""
            pages.append(clean_text(raw))
    return "\n\n".join(pages)

def chunk_text(text: str, size: int = 3000) -> List[str]:
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= size:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks

def get_semantic_chunks(text: str) -> List[str]:
    if os.path.exists(CHUNK_CACHE_FILE):
        with open(CHUNK_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [item["chunk"] for item in data]
    chunks = chunk_text(text)
    with open(CHUNK_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump([{"chunk": c} for c in chunks], f, ensure_ascii=False, indent=2)
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    global embed_cache
    if not embed_cache and os.path.exists(EMBED_CACHE_FILE):
        with open(EMBED_CACHE_FILE, "rb") as f:
            embed_cache = pickle.load(f)
    missing = [t for t in texts if t not in embed_cache]
    if missing:
        resp = client.embeddings.create(model=EMBED_MODEL, input=missing)
        for t, d in zip(missing, resp.data):
            embed_cache[t] = d.embedding
        with open(EMBED_CACHE_FILE, "wb") as f:
            pickle.dump(embed_cache, f)
    return [embed_cache[t] for t in texts]

def build_faiss_index(embs: List[List[float]]) -> faiss.IndexFlatIP:
    arr = np.array(embs, dtype="float32")
    faiss.normalize_L2(arr)
    dim = arr.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(arr)
    return index

def answer_query(query: str, k: int = 3) -> str:
    if idx is None or status["status"] != "ready":
        raise RuntimeError("Contexto não está pronto. Rode POST /process antes.")
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    q_vec = np.array([q_emb], dtype="float32")
    faiss.normalize_L2(q_vec)
    _, I = idx.search(q_vec, k)
    context = "\n\n---\n\n".join(blocos[i] for i in I[0])
    key = json.dumps({"ctx": context, "q": query}, ensure_ascii=False)
    if key in chat_cache:
        return chat_cache[key]
    prompt = [
        {"role": "system", "content": "Você é um assistente que responde com base no contexto."},
        {"role": "user",   "content": f"Contexto:\n{context}\n\nPergunta: {query}"}
    ]
    chat = client.chat.completions.create(model=CHAT_MODEL, messages=prompt, temperature=0.2)
    out = chat.choices[0].message.content
    chat_cache[key] = out
    with open(CHAT_CACHE_FILE, "wb") as f:
        pickle.dump(chat_cache, f)
    return out

# --- Extrator de site para PDF ---

def extrair_texto(url: str) -> str:
    r = requests.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')
    for t in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        t.decompose()
    texto = "\n".join(l.strip() for l in soup.get_text().splitlines() if l.strip())

    # Remover caracteres fora do padrão Unicode básico
    texto = texto.encode('latin-1', errors='ignore').decode('latin-1')

    return texto


def gerar_pdf(urls: List[str], output: str):
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_font('DejaVu', '', os.path.join(FONTS_DIR, 'DejaVuSans.ttf'), uni=True)
    pdf.add_font('DejaVu', 'B', os.path.join(FONTS_DIR, 'DejaVuSans-Bold.ttf'), uni=True)
    for url in urls:
        txt = extrair_texto(url)
        pdf.add_page()
        pdf.set_font('DejaVu', 'B', 14)
        pdf.multi_cell(0, 8, f"Conteúdo de: {url}")
        pdf.ln(2)
        pdf.set_font('DejaVu', '', 12)
        pdf.multi_cell(0, 6, txt)
    pdf.output(output)

# --- Thread de processamento ---

def process_urls(urls: List[str]):
    global idx, blocos, status, embed_cache

    # — Limpa caches de chunks e embeddings pra forçar atualização
    try:
        if os.path.exists(CHUNK_CACHE_FILE):
            os.remove(CHUNK_CACHE_FILE)
        if os.path.exists(EMBED_CACHE_FILE):
            os.remove(EMBED_CACHE_FILE)
        embed_cache.clear()
    except Exception as e:
        logger.warning("Não foi possível limpar caches: %s", e)

    try:
        status.update(status="queued", progress=0, message=None)

        status.update(status="generating_pdf", progress=10)
        gerar_pdf(urls, DEFAULT_PDF_PATH)

        status.update(status="reading_pdf", progress=30)
        text = load_pdf_clean(DEFAULT_PDF_PATH)

        status.update(status="chunking", progress=50)
        blocos = get_semantic_chunks(text)

        status.update(status="embedding", progress=70)
        embs = get_embeddings(blocos)

        status.update(status="indexing", progress=85)
        idx = build_faiss_index(embs)

        status.update(status="ready", progress=100)
    except Exception as e:
        logger.exception("Erro no processamento")
        status.update(status="error", progress=0, message=str(e))


# --- Flask + Swagger setup ---

app = Flask(__name__)
api = Api(app, version="1.0", title="Unified PDF RAG API",
          description="Contexto só é reconstruído em POST /process", doc="/docs")

process_model = api.model("Process", {
    "urls": fields.List(fields.String, required=True, description="URLs para processar")
})
query_model = api.model("Query", {
    "query": fields.String(required=True, description="Pergunta"),
    "k":     fields.Integer(default=3,   description="Número de chunks")
})
status_model = api.model("Status", {
    "status":   fields.String(required=True),
    "progress": fields.Integer(required=True),
    "message":  fields.String
})

@api.route("/process")
class ProcessResource(Resource):
    @api.expect(process_model)
    def post(self):
        """Recria todo o contexto (uso de cache só aqui)."""
        data = request.get_json()
        threading.Thread(target=process_urls, args=(data["urls"],), daemon=True).start()
        return {"message": "Processamento iniciado"}, 202

@api.route("/status")
class StatusResource(Resource):
    @api.marshal_with(status_model)
    def get(self):
        """Retorna status atual."""
        return status

@api.route("/query")
class QueryResource(Resource):
    @api.expect(query_model)
    def post(self):
        """Responde usando contexto anterior ou erro se não pronto."""
        data = request.get_json()
        try:
            resp = answer_query(data["query"], data.get("k", 3))
            return {"response": resp}
        except RuntimeError as e:
            api.abort(400, str(e))

if __name__ == "__main__":
    os.makedirs(FONTS_DIR, exist_ok=True)
    # Carrega caches existentes (sem reconstruir)
    if os.path.exists(CHUNK_CACHE_FILE):
        with open(CHUNK_CACHE_FILE, "r", encoding="utf-8") as f:
            blocos = [ch["chunk"] for ch in json.load(f)]
    if os.path.exists(EMBED_CACHE_FILE):
        with open(EMBED_CACHE_FILE, "rb") as f:
            embed_cache = pickle.load(f)
    if os.path.exists(CHAT_CACHE_FILE):
        with open(CHAT_CACHE_FILE, "rb") as f:
            chat_cache = pickle.load(f)
    app.run(host="0.0.0.0", port=8001, debug=True)