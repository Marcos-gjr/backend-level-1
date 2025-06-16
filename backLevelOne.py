import os
import re
import pdfplumber
import numpy as np
import faiss
from openai import OpenAI
from flask import Flask, request, jsonify

client = OpenAI(api_key=os.getenv("OPENAI_API"))
DIMENSIONS = 3072

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def load_pdf(path):
    with pdfplumber.open(path) as pdf:
        return clean_text(" ".join(p.extract_text() or "" for p in pdf.pages))

def split_chunks(text, max_tokens=500):
    words = text.split()
    chunks, chunk = [], []
    count = 0
    for word in words:
        chunk.append(word)
        count += 1
        if count >= max_tokens:
            chunks.append(" ".join(chunk))
            chunk, count = [], 0
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def get_embeddings(texts):
    return [client.embeddings.create(input=[t], model="text-embedding-3-large").data[0].embedding for t in texts]

def build_index(embs):
    index = faiss.IndexFlatIP(DIMENSIONS)
    arr = np.array(embs).astype("float32")
    faiss.normalize_L2(arr)
    index.add(arr)
    return index

app = Flask(__name__)
texto = load_pdf("./dockermanual.pdf")
chunks = split_chunks(texto)
embeddings = get_embeddings(chunks)
index = build_index(embeddings)

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    q = data.get("query")
    q_emb = client.embeddings.create(input=[q], model="text-embedding-3-large").data[0].embedding
    vec = np.array([q_emb], dtype="float32")
    faiss.normalize_L2(vec)
    _, I = index.search(vec, 3)
    ctx = "\n\n---\n\n".join(chunks[i] for i in I[0])
    return jsonify({"contexto": ctx})

if __name__ == "__main__":
    app.run(debug=True, port=8001)
