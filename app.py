import os
import re
import json
import pickle
import logging
import threading
import requests

from typing import List, Optional, Tuple
from dotenv import load_dotenv
from flask import Flask, request
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
from fpdf import FPDF
import pdfplumber
import faiss
import numpy as np
from openai import OpenAI
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API = os.getenv("OPENAI_API")
logger.info("OPENAI_API_KEY 3 de %s", OPENAI_API)
client = OpenAI(api_key=OPENAI_API)

CHUNK_MODEL      = "gpt-4"
EMBED_MODEL      = "text-embedding-3-large"
CHAT_MODEL       = "gpt-3.5-turbo"
DEFAULT_PDF_PATH = "./documentacao.pdf"
CHUNK_CACHE_FILE = "semantic_chunks.json"
EMBED_CACHE_FILE = "embeddings_cache.pkl"
CHAT_CACHE_FILE  = "chat_cache.pkl"
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
DEPLOY_DIR = os.getcwd()

itens = os.listdir(DEPLOY_DIR)
logger.info("Conteúdo de %s: %s", DEPLOY_DIR, itens)

FONTS_DIR = Path('./fonts')
logger.info("Usando pasta de fontes em: %s", FONTS_DIR)

# Status do process
status = {
    # status que aparecerão durante o processoidle, queued, generating_pdf, reading_pdf, chunking, embedding, indexing, ready, error
    "status": "idle",    
    "progress": 0,
    "message": None
}

# Contexto inicial do RAG 
embed_cache: dict = {}
chat_cache: dict  = {}
idx: Optional[faiss.IndexFlatIP] = None
blocos: Optional[List[str]] = None


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
    global blocos
    blocos = get_semantic_chunks(query)

    if idx is None or status["status"] != "ready":
        raise RuntimeError("Contexto não está pronto. Rode POST /process antes.")

    q_emb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    q_vec = np.array([q_emb], dtype="float32")
    faiss.normalize_L2(q_vec)

    _, I = idx.search(q_vec, k)
    try:
        i0 = int(I[0][0])
    except Exception:
        i0 = 0
    if not blocos or i0 < 0 or i0 >= len(blocos):
        i0 = 0
    context = blocos[i0]

    key = json.dumps({"ctx": context, "q": query}, ensure_ascii=False)
    prompt = [
        {
            "role": "system",
            "content": (
                "Você é um assistente que segue estas regras:\n\n"
                "1. Se o usuário fizer perguntas sobre suas próprias capacidades ou uso da ferramenta "
                "(ex.: “O que você pode me ajudar?”, “Como funciona este chat?”), responda normalmente, "
                "explicando suas funções e limitações.\n"
                "2. Para qualquer outra pergunta técnica ou de domínio, responda **apenas** com base no "
                "contexto que o usuário **explicitamente** forneceu nesta conversa.\n"
                "3. Se o contexto **não** contiver informação suficiente para responder, responda **exatamente** "
                "“Não sei ainda.”\n"
                "4. Cuidado ao elaborar com conteúdo além do contexto.\n\n"
                "---\n\n"
                "**Exemplos de comportamento**\n\n"
                "**Incorreto:**\n"
                "Usuário: o que você pode me ajudar?\n"
                "Assistente: Não sei ainda.\n\n"
                "**Correto:**\n"
                "Usuário: o que você pode me ajudar?\n"
                "Assistente: Eu posso te ajudar a entender e utilizar o Docker Compose, ...\n\n"
                "**Correto (domínio sem contexto):**\n"
                "Usuário: Como usar variáveis de ambiente e arquivos .env?\n"
                "Assistente: Não sei ainda.\n"
            )
        },
        {
            "role": "user",
            "content": f"Contexto:\n{context}\n\nPergunta: {query}"
        }
    ]

    chat = client.chat.completions.create(model=CHAT_MODEL, messages=prompt, temperature=0.2)
    out = chat.choices[0].message.content
    chat_cache[key] = out
    with open(CHAT_CACHE_FILE, "wb") as f:
        pickle.dump(chat_cache, f)

    app.logger.info("Contexto: %s", context)
    app.logger.info("Pergunta: %s", query)
    app.logger.info("Resposta: %s", out)

    return out

def extrair_texto(url: str) -> str:
    r = requests.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')
    for t in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        t.decompose()
    texto = "\n".join(l.strip() for l in soup.get_text().splitlines() if l.strip())
    return texto.encode('latin-1', errors='ignore').decode('latin-1')

def gerar_pdf(urls: List[str], output: str):
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_font('DejaVu', '', str(FONTS_DIR / 'DejaVuSans.ttf'), uni=True)
    pdf.add_font('DejaVu', 'B', str(FONTS_DIR / 'DejaVuSans-Bold.ttf'), uni=True)
    for url in urls:
        txt = extrair_texto(url)
        pdf.add_page()
        pdf.set_font('DejaVu', 'B', 14)
        pdf.multi_cell(0, 8, f"Conteúdo de: {url}")
        pdf.ln(2)
        pdf.set_font('DejaVu', '', 12)
        pdf.multi_cell(0, 6, txt)
    pdf.output(output)

def process_urls(urls: List[str], files_data: Optional[List[Tuple[str, bytes]]] = None):
    global idx, blocos, status, embed_cache
    try:
        for f in (CHUNK_CACHE_FILE, EMBED_CACHE_FILE):
            if os.path.exists(f):
                os.remove(f)
        embed_cache.clear()
    except Exception as e:
        logger.warning("Não foi possível limpar caches: %s", e)

    try:
        status.update(status="queued", progress=0, message=None)

        texts = []
        if urls:
            status.update(status="generating_pdf", progress=10)
            gerar_pdf(urls, DEFAULT_PDF_PATH)
            status.update(status="reading_pdf", progress=30)
            texts.append(load_pdf_clean(DEFAULT_PDF_PATH))

        if files_data:
            for filename, blob in files_data:
                safe = secure_filename(filename)
                path = os.path.join(DEPLOY_DIR, safe)
                with open(path, "wb") as f_out:
                    f_out.write(blob)
                status.update(status="reading_pdf", progress=40)
                texts.append(load_pdf_clean(path))

        full_text = "\n\n".join(texts)
        status.update(status="chunking", progress=50)
        blocos = get_semantic_chunks(full_text)
        status.update(status="embedding", progress=70)
        embs = get_embeddings(blocos)
        status.update(status="indexing", progress=85)
        idx = build_faiss_index(embs)
        status.update(status="ready", progress=100)

    except Exception as e:
        logger.exception("Erro no processamento")
        status.update(status="error", progress=0, message=str(e))

app = Flask(__name__)
api = Api(app, version="1.0", title="Unified PDF RAG API",
          description="Contexto só é reconstruído em POST /process", doc="/docs")

app.logger.handlers = logger.handlers
app.logger.setLevel(logger.level)

upload_parser = reqparse.RequestParser()
upload_parser.add_argument("urls", type=str, required=False, location="form", help="JSON com lista de URLs")
upload_parser.add_argument("files", type='file', location="files", required=False, action="append", help="Arquivos PDF")

status_model = api.model("Status", {"status": fields.String, "progress": fields.Integer, "message": fields.String})
query_model  = api.model("Query", {"query": fields.String(required=True), "k": fields.Integer(default=3)})

@api.route("/process")
class ProcessResource(Resource):
    @api.expect(upload_parser)
    def post(self):
        urls = []
        if request.form.get("urls"):
            try:
                urls = json.loads(request.form["urls"])
            except json.JSONDecodeError:
                api.abort(400, "Campo 'urls' não é um JSON válido")

        files_data: List[Tuple[str, bytes]] = []
        for f in request.files.getlist("files"):
            name = secure_filename(f.filename)
            blob = f.read()
            files_data.append((name, blob))

        threading.Thread(target=process_urls, args=(urls, files_data), daemon=True).start()
        return {"message": "Processamento iniciado"}, 202

@api.route("/status")
class StatusResource(Resource):
    @api.marshal_with(status_model)
    def get(self):
        return status

@api.route("/query")
class QueryResource(Resource):
    @api.expect(query_model)
    def post(self):
        data = request.get_json()
        try:
            answer = answer_query(data["query"], data.get("k", 3))
            return {"answer": answer}
        except RuntimeError as e:
            api.abort(400, str(e))

if __name__ == "__main__":
    if os.path.exists(CHUNK_CACHE_FILE):
        with open(CHUNK_CACHE_FILE, "r", encoding="utf-8") as f:
            blocos = [c["chunk"] for c in json.load(f)]
    if os.path.exists(EMBED_CACHE_FILE):
        with open(EMBED_CACHE_FILE, "rb") as f:
            embed_cache = pickle.load(f)
    if os.path.exists(CHAT_CACHE_FILE):
        with open(CHAT_CACHE_FILE, "rb") as f:
            chat_cache = pickle.load(f)

    app.run(host="0.0.0.0", port=8001, debug=True)