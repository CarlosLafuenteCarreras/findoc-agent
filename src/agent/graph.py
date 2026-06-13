from langchain_openai import ChatOpenAI
import os
from src.agent.tools import search_financial_documents, search_regulation, extract_financial_metric, compare_entities, detect_risk_signals 
from langchain.agents import create_react_agent, AgentExecutor
from src.config import config
from langchain import hub


def build_agent() -> AgentExecutor:
    
    
    llm = ChatOpenAI(api_key= os.getenv("OPENAI_API_KEY"), model_name=config.LLM_MODEL, temperature= config.TEMPERATURE)
    
    tools = [
        search_financial_documents,
        search_regulation,
        extract_financial_metric,
        compare_entities,
        detect_risk_signals
    ]
    
    prompt = hub.pull("hwchase17/react")
    
    agent = create_react_agent(llm, tools, prompt=prompt)
    
    return AgentExecutor(agent=agent, tools = tools, verbose= True, max_iterations= 7, handle_parsing_errors=True)