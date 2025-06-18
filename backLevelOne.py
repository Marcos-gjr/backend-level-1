import os
import re
import json
import pickle
import pdfplumber
import faiss
import tiktoken
import numpy as np
from openai import OpenAI
from typing import List

from flask import Flask, request
from flask_restx import Api, Resource, fields

OPENAI_API_KEY = os.getenv("OPENAI_API")

client = OpenAI(api_key=OPENAI_API_KEY)

CHUNK_MODEL = os.getenv("CHUNK_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
CHAT_MODEL  = os.getenv("CHAT_MODEL")

DIMENSIONS = os.getenv("DIMENSIONS")

tokenizer = tiktoken.get_encoding("cl100k_base")
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def clean_text(text: str) -> str:
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_pdf_clean(path: str) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            raw = p.extract_text() or ""
            pages.append(clean_text(raw))
    return "\n\n".join(pages)


CHUNK_CACHE_FILE = "semantic_chunks.txt"
def get_semantic_chunks(text: str) -> List[str]:
    with open(CHUNK_CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["chunk"] for item in data]

EMBED_CACHE_FILE = "embeddings_cache.pkl"
with open(EMBED_CACHE_FILE, "rb") as f:
    embed_cache = pickle.load(f)

def get_embeddings(texts: List[str]) -> List[List[float]]:
    missing = [t for t in texts if t not in embed_cache]
    if missing:
        raise ValueError(f"Faltam embeddings para {len(missing)} chunks")
    return [embed_cache[t] for t in texts]

def build_faiss_index(embs: List[List[float]]) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(DIMENSIONS)
    arr   = np.array(embs, dtype="float32")
    faiss.normalize_L2(arr)
    index.add(arr)
    return index

CHAT_CACHE_FILE = "chat_cache.pkl"
try:
    with open(CHAT_CACHE_FILE, "rb") as f:
        chat_cache = pickle.load(f)
except FileNotFoundError:
    chat_cache = {}

def answer_query(query: str, index, chunks: List[str], k: int = 3) -> str:


    q_emb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    q_vec = np.array([q_emb], dtype="float32")
    faiss.normalize_L2(q_vec)
    _, I = index.search(q_vec, k)

    context = "\n\n---\n\n".join(chunks[i] for i in I[0])
    print(f"Contexto selecionado (top {k} chunks):\n{context}\n")

    cache_key = json.dumps({"ctx": context, "q": query}, ensure_ascii=False)
    if cache_key in chat_cache:
        print("[CHAT] resposta obtida do cache")
        return chat_cache[cache_key]

    prompt = [
        {"role":"system","content":"Você é um assistente que responde com base no contexto."},
        {"role":"user",  "content":f"Contexto:\n{context}\n\nPergunta: {query}"}
    ]
    

    chat = client.chat.completions.create(model=CHAT_MODEL, messages=prompt, temperature=0.2)
    out = chat.choices[0].message.content


    chat_cache[cache_key] = out
    with open(CHAT_CACHE_FILE, "wb") as f:
        pickle.dump(chat_cache, f)

    return out

texto  = load_pdf_clean("./dockermanual.pdf")
blocos = get_semantic_chunks(texto)
embs   = get_embeddings(blocos)
idx    = build_faiss_index(embs)

app = Flask(__name__)
api = Api(
    app,
    version="1.0",
    title="API para busca de PDF via RAg",
    description="API para testar as vbuscas via RAG em PDF com utilização do FAISS",
    doc="/docs"
)

query_model = api.model("Query", {
    "query": fields.String(required=True, description="Texto da pergunta"),
    "k":     fields.Integer(default=3, description="Chunks para serem retornados")
})

@api.route("/query")
class QueryResource(Resource):
    @api.expect(query_model)
    def post(self):
        """Faça uma pergunta baseada no conteúdo do PDF."""
        data = request.get_json()
        q = data.get("query")
        k = data.get("k", 3)
        resposta = answer_query(q, idx, blocos, k)
        return {"response": resposta}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8001)
