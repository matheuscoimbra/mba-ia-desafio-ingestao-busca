import os
import re
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

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
                #print(f"Faturamento da empresa: {empresa.nome}")
                emb_doc.append(Document(page_content=json.dumps(empresa.__dict__, ensure_ascii=False, indent=2)))
            else:
                print(f"Não foi possível fazer o parse da linha: {line}")

    store.add_documents(documents=emb_doc)


if __name__ == "__main__":
    ingest_pdf()