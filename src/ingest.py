import os
import re
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.empresa import Empresa

load_dotenv()

padrao = re.compile(r"^(.+?)(\s+)?R\$[\s]*([\d\.,]+)\s+(\d{4})$")
PDF_PATH = os.getenv("PDF_PATH")
current_dir = Path(__file__).parent
embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL","text-embedding-3-small"))

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
    connection=os.getenv("DATABASE_URL"),
    use_jsonb=True,
)


def ingest_pdf2():
    doc = PyPDFLoader(str(current_dir.parent / PDF_PATH))
    file = doc.load()

    emb_doc = []
    for c in file:
        for line in c.page_content.split("\n"):
            match = padrao.match(line)
            if match:
                descricao,_, valor, ano  = match.groups()
                empresa = Empresa(
                    nome=descricao,
                    faturamento="R$ " + valor,
                    ano_fundacao=int(ano),
                )
                emb_doc.append(Document(page_content=json.dumps(empresa.__dict__, ensure_ascii=False, indent=2)))
            else:
                print(f"Não foi possível fazer o parse da linha: {line}")

    store.add_documents(documents=emb_doc)

def ingest_pdf():
    doc = PyPDFLoader(str(current_dir.parent / PDF_PATH))
    file = doc.load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150, add_start_index=False).split_documents(file)
    if not splits:
        raise SystemExit(0)

    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in splits
    ]

    ids = [f"doc-{i}" for i in range(len(enriched))]

    store.add_documents(documents=enriched, ids=ids)


if __name__ == "__main__":
    ingest_pdf()
