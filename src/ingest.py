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


def ingest_pdf():
    doc = PyPDFLoader(str(current_dir / PDF_PATH))
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