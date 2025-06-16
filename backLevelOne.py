import re
import pdfplumber
from flask import Flask, request, jsonify

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def load_pdf(path):
    with pdfplumber.open(path) as pdf:
        return " ".join(clean_text(p.extract_text() or "") for p in pdf.pages)

app = Flask(__name__)
texto = load_pdf("./dockermanual.pdf")

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    termo = data.get("query", "")
    ocorrencias = len(re.findall(re.escape(termo), texto, re.IGNORECASE))
    return jsonify({"ocorrencias": ocorrencias})

if __name__ == "__main__":
    app.run(debug=True, port=8001)
