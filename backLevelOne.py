import os
import re
import json
import pdfplumber
import numpy as np
from openai import OpenAI
from flask import Flask, request, jsonify

client = OpenAI(api_key=os.getenv("OPENAI_API"))

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

def search(query_emb, embeddings, k=3):
    sims = [(i, np.dot(query_emb, emb)) for i, emb in enumerate(embeddings)]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in sims[:k]]

app = Flask(__name__)
texto = load_pdf("./dockermanual.pdf")
chunks = split_chunks(texto)
embeddings = get_embeddings(chunks)

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    q = data.get("query")
    q_emb = client.embeddings.create(input=[q], model="text-embedding-3-large").data[0].embedding
    top = search(q_emb, embeddings)
    ctx = "\n\n---\n\n".join(chunks[i] for i in top)
    return jsonify({"contexto": ctx})

if __name__ == "__main__":
    app.run(debug=True, port=8001)
