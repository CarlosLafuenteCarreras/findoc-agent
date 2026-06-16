from langchain_openai import ChatOpenAI
import os

from src.agent.tools import search_financial_documents, search_regulation, extract_financial_metric, compare_entities, detect_risk_signals 

from langgraph.prebuilt import create_react_agent

from src.config import config

from langchain_core.messages import SystemMessage


SYSTEM_PROMPT = """Eres un experto en análisis financiero y normativa bancaria.

Tienes acceso a dos bases de conocimiento:
- Informes financieros: informes anuales y Pillar 3 de BBVA, Santander y CaixaBank
- Normativa regulatoria: Basilea III, CRR, IFRS 9 y EBA Guidelines

Reglas:
- Para preguntas sobre resultados, métricas o datos de empresas → usa search_financial_documents o extract_financial_metric
- Para preguntas sobre normativa, regulación o requisitos → usa search_regulation
- Para comparar dos entidades → usa compare_entities
- Para detectar riesgos en informes → usa detect_risk_signals
- Responde siempre en el mismo idioma que la pregunta
- Nunca inventes datos financieros ni artículos normativos
- Cita siempre la fuente y página"""


def build_agent():
    
    
    llm = ChatOpenAI(api_key= os.getenv("OPENAI_API_KEY"), model_name=config.LLM_MODEL, temperature= config.TEMPERATURE)
    
    tools = [
        search_financial_documents,
        search_regulation,
        extract_financial_metric,
        compare_entities,
        detect_risk_signals
    ]
    
    prompt = SYSTEM_PROMPT
    
    return create_react_agent(model=llm, tools= tools, prompt=SYSTEM_PROMPT)