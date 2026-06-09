from langchain_openai import OpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from src.config import config
import os


FINANCIAL_PROMPT = """Eres un analista financiero experto. Responde a la pregunta ÚNICAMENTE usando la información extraida de informes financieros reales. 

No hagas suposiciones ni inventes información. Si no sabes la respuesta o la información no está en el contexto di: "No encuentro esta información en los documentos proporcionados."

Contexto:
{context}

Pregunta: {question}

Respuesta (cita la entidad, año y página cuando sea posible):"""



REGULATORY_PROMPT = """Eres un experto en regulación financiera bancaria europea. Responde la pregunta usando ÚNICAMENTE la información del siguiente contexto extraído de normativa regulatoria oficial.

Si la información no está en el contexto, di: "No encuentro esta disposición en la normativa proporcionada."
Indica siempre la normativa fuente (CRR, IFRS 9, Basilea III, EBA Guidelines) y el artículo si aparece.

Contexto:
{context}

Pregunta: {question}

Respuesta (cita la normativa, artículo y página):"""


def financial_chain_builder(vectorstore: Chroma) -> RetrievalQA:
    """
    Build RAG chain for financial queries

    Args:
        vectorstore (Chroma): Chroma vector store with financial documents

    Returns:
        RetrievalQA: RAG chain for financial queries
    """
    llm = OpenAI(model_name=config.LLM_MODEL, temperature=config.TEMPERATURE, api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt= PromptTemplate.from_template(template= FINANCIAL_PROMPT)
    
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(search_type= "similarity", search_kwargs={"k": config.TOP_K_FINANCIAL}), return_source_documents=True, chain_type_kwargs={"prompt": prompt})


def regulatory_chain_builder(vectorstore: Chroma) -> RetrievalQA:
    """
    Build RAG chain for regulatory queries

    Args:
        vectorstore (Chroma): Chroma vector store with regulatory documents
    Returns:
        RetrievalQA: RAG chain for regulatory queries
    """
    llm = OpenAI(model_name=config.LLM_MODEL, temperature=config.TEMPERATURE, api_key=os.getenv("OPENAI_API_KEY"))      
    
    prompt= PromptTemplate.from_template(template= REGULATORY_PROMPT)
    
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(search_type= "similarity", search_kwargs={"k": config.TOP_K_REGULATORY}), return_source_documents=True, chain_type_kwargs={"prompt": prompt})

def query_chain(chain: RetrievalQA, question: str) -> dict:
    """
    Query any of both RAG chains and return the answer and sources

    Args:
        chain (RetrievalQA): RAG chain to query
        question (str): User question

    Returns:
        dict: Dictionary with 'answer' and 'sources'
    """
    
    result = chain.invoke({"query": question})
    sources = []
    
    for doc in result.get("source_documents", []):
        sources.append({
            "content": doc.page_content[:500],
            "page": doc.metadata.get("page"),
            "collection":doc.metadata.get("collection"),
            "source": doc.metadata.get("source"),
            "entity": doc.metadata.get("entity"),
            "regulation": doc.metadata.get("regulation"),
            "subtype": doc.metadata.get("subtype"),
            "year": doc.metadata.get("year")
        })
    return {"answer": result['result'], "sources": sources}
