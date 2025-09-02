import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

model = ChatOpenAI(model="gpt-5-mini", temperature=0.1)

embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL","text-embedding-3-small"))

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
    connection=os.getenv("DATABASE_URL"),
    use_jsonb=True,
)


def transform_results(results)->str:
    return f"\n-".join(k.page_content for k, v in results)


def search_prompt(pergunta=None):
    results = store.similarity_search_with_score(pergunta, k=10)

    result_list_format = transform_results(results)


    template = PromptTemplate(
         input_variables=["contexto", "pergunta"],
         template=PROMPT_TEMPLATE
    )

    #print(template.format(contexto=result_list_format, pergunta=pergunta))
    pipeline = template | model | StrOutputParser()
    result = pipeline.invoke({"contexto": result_list_format, "pergunta": pergunta})
    return result

