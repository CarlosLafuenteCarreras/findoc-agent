from langchain.tools import tool
# decorador para marcar funciones como tools
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from src.config import config
from typing import Optional
import os

# Vectorstores inicializados, _ porque son de uso interno
# Optional indica que puede ser None hasta que se inicialicen con la función initialize_tools
_financial_vs: Optional[Chroma] = None
_regulatory_vs: Optional[Chroma] = None



def initialize_tools(financial_vs: Chroma, regulatory_vs: Chroma):
    global _financial_vs, _regulatory_vs # Indica que son las definidas antes
    _financial_vs = financial_vs
    _regulatory_vs = regulatory_vs
    
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=config.LLM_MODEL,openai_api_key=config.OPENAI_API_KEY, temperature=config.TEMPERATURE)


@tool
def search_financial_documents(query: str) -> str:
    """
    Busca información en los informes financieros indexados (informes anuales, Pillar 3).
    Úsala para preguntas sobre resultados financieros, métricas de negocio, estrategia,
    mercados, empleados, o cualquier dato concreto de una empresa o banco específico.
    Input: pregunta o término de búsqueda en lenguaje natural.
    Output: fragmentos relevantes con entidad, año y página.
    """
    
    # sacamos topk de similaridad
    results = _financial_vs.similarity_search(query, k=config.TOP_K_FINANCIAL)
    
    if not results:
        return "No se encontraron documentos financieros relevantes."
    
    formatted = []
    for doc in results:
        header = f"[{doc.metadata.get('company', '?')} — {doc.metadata.get('type', '?')} {doc.metadata.get('year', '?')} — Pág. {doc.metadata.get('page', '?')}]"
        formatted.append(f"{header}\n{doc.page_content[:600]}")
        
    return "\n\n".join(formatted)

@tool
def search_regulation(query:str, regulation: str = None) -> str:
    """
    Busca en la normativa regulatoria bancaria indexada (CRR, CRD, IFRS 9, Basilea III, EBA Guidelines).
    Úsala para preguntas sobre requisitos regulatorios, definiciones normativas, metodologías IRB,
    tratamiento contable, requisitos de capital, o cualquier pregunta sobre normativa bancaria.
    Input: pregunta regulatoria y opcionalmente el reglamento específico (CRR, IFRS9, Basilea3, EBA_GL).
    Output: fragmentos de normativa con referencia de artículo y página.
    """
    filter_dict={}
    
    if regulation:
        filter_dict["regulation"] = regulation.upper()

    results = _regulatory_vs.similarity_search(query, k=config.TOP_K_REGULATORY, filter=filter_dict if filter_dict else None)
    
    if not results:
        return "No se encontraron documentos regulatorios relevantes."
    
    formatted = []
    
    for doc in results:
        header = f"[{doc.metadata.get('regulation', '?')} — {doc.metadata.get('subtype', '?')} {doc.metadata.get('year', '?')} — Pág. {doc.metadata.get('page', '?')}]"
        formatted.append(f"{header}\n{doc.page_content[:600]}")
        
    return "\n\n".join(formatted)

@tool
def extract_financial_metric(metric_name: str, entity: str = None, year: str =None) -> str:
    """
    Extrae el valor exacto de una métrica financiera de los documentos indexados.
    Úsala cuando el usuario pida cifras concretas: beneficio neto, EBITDA, ingresos, ratio CET1, tasa de mora, PD media, LGD, número de empleados, dividendo por acción.
    Input: nombre de la métrica, y opcionalmente entidad y año para filtrar.
    Output: valor exacto con fuente y página.
    """
    query = f"Extrae el valor de {metric_name} con la cifra exacta. Si hay varias menciones, prioriza la más reciente y la que tenga cifra concreta. Devuelve solo la cifra, sin texto adicional."
    
    filter_dict={}
    if entity:
        filter_dict["entity"] = entity
    if year:
        filter_dict["year"] = year
    
    results = _financial_vs.similarity_search(query, k=config.TOP_K_REGULATORY, filter=filter_dict if filter_dict else None)
    
    context = "\n".join([doc.page_content for doc in results])
    
    extraction_prompt = f"""Del siguiente texto financiero extrae el valor exacto de '{metric_name}'.
    {'Entidad: ' + entity if entity else ''}
    {'Año: ' + year if year else ''}

    Responde SOLO con: [métrica]: [valor] ([año]) — Fuente: [nombre archivo] Pág. [número]
    Si no aparece la métrica: "Métrica '{metric_name}' no encontrada en los documentos."

    Texto:
    {context}"""

    respuesta = _get_llm().invoke(extraction_prompt)
    return respuesta


@tool
def compare_entities(metric_name: str, entity1: str, entity2: str, year: str) -> str:
    """
    Compara el valor de una métrica financiera entre dos entidades (empresas o bancos) para un año específico.
    Úsala para preguntas comparativas: ¿Qué banco tuvo mayor beneficio neto en 2023? ¿Cómo evolucionó la tasa de mora de Santander vs BBVA en 2022?
    Input: nombre de la métrica, dos entidades a comparar, y año.
    Output: comparación clara indicando qué entidad tiene mejor resultado y las cifras exactas.
    """
    query = f"Compara el valor de {metric_name} entre {entity1} y {entity2} para el año {year}. Devuelve solo la comparación con cifras exactas y sin texto adicional."
    
    filter1 = {"entity": entity1}
    filter2 = {"entity": entity2}
    
    if year:
        filter1["year"] = year
        filter2["year"] = year
    
    results_entity1 = _financial_vs.similarity_search(query, k=config.TOP_K_FINANCIAL, filter=filter1)
    results_entity2 = _financial_vs.similarity_search(query, k=config.TOP_K_FINANCIAL, filter=filter2)
    
    context_entity1 = "\n".join([doc.page_content for doc in results_entity1])
    context_entity2 = "\n".join([doc.page_content for doc in results_entity2])
    
    comparation_prompt = f"""Compara el valor de '{metric_name}' entre '{entity1}' y '{entity2}' para el año {year if year else 'no especificado'}.

    Contexto para {entity1}:
    {context_entity1}

    Contexto para {entity2}:
    {context_entity2}
    
    Responde con una tabla comparativa indicando claramente los valores de cada entidad tanto en porcentaje absoluto como porcentual. Cita la fuente y pagina de cada cifra. Si no se encuentra la métrica para alguna entidad, indícalo claramente.
    """

    respuesta = _get_llm().invoke(comparation_prompt)
    return respuesta

@tool
def detect_risk_signals(entity: str = None, doc_type: str = None) -> str:
    """
    Detecta y lista señales de riesgo mencionadas en los informes financieros.
    Úsala cuando el usuario pregunte por riesgos, advertencias, litigios, deterioros,
    cambios regulatorios negativos o cualquier señal de alerta en los documentos.
    Input: opcionalmente entidad y tipo de documento para filtrar.
    Output: lista de señales de riesgo encontradas con fuente y página.
    """
    query = "riesgo advertencia litigio deterioro pérdida incumplimiento alerta"

    filter_dict = {}
    if entity:
        filter_dict["entity"] = entity
    if doc_type:
        filter_dict["doc_type"] = doc_type

    results = _financial_vs.similarity_search( query, k=config.TOP_K_FINANCIAL, filter=filter_dict if filter_dict else None)

    context = "\n\n".join([
        f"[{d.metadata.get('entity')} — Pág. {d.metadata.get('page')}]\n{d.page_content[:500]}"
        for d in results
    ])

    risk_prompt = f"""Del siguiente texto de informes financieros, identifica y lista todas las señales de riesgo:
    litigios pendientes, deterioros de activos, advertencias de auditoría, cambios regulatorios adversos, concentraciones de riesgo, exposiciones problemáticas, o cualquier lenguaje de cautela.

    Organiza por categoría de riesgo (operacional, crédito, mercado, regulatorio, reputacional).
    Cita la fuente y página de cada señal.

    Texto:
    {context}"""

    response = _get_llm().invoke(risk_prompt)
    return response.content