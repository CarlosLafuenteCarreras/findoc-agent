import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import json
import logging
logging.basicConfig(level=logging.WARNING)

from src.embeddings.embedder import load_financial_vectorstore, load_regulatory_vectorstore
from src.retrieval.retriever import financial_chain_builder, regulatory_chain_builder, query_chain


financial_vs= load_financial_vectorstore()
regulatory_vs= load_regulatory_vectorstore()

financial_chain= financial_chain_builder(financial_vs)
regulatory_chain= regulatory_chain_builder(regulatory_vs)

print("Colecciones cargadas, chains construidas")

with open("tests/eval_questions.json","r",encoding="utf-8") as f:
    eval_questions=json.load(f)

# Test colección financiera
print("\n" + "="*60)
print("EVALUACIÓN COLECCIÓN FINANCIERA")
print("=" * 60)


for i, item in enumerate(eval_questions["financial"], 1):
    print(f"\n[{i}] {item['question']}")
    print(f"    Tipo esperado: {item['expected']}")
    print("-" * 40)
    result = query_chain(financial_chain, item["question"])
    print(f"Respuesta:\n{result['answer']}")
    print("\nFuentes:")
    for src in result["sources"]:
        print(f"  • {src['source']}")

# Test colección regulatoria
print("\n" + "=" * 60)
print("EVALUACIÓN COLECCIÓN REGULATORIA")
print("=" * 60)

for i, item in enumerate(eval_questions["regulatory"], 1):
    print(f"\n[{i}] {item['question']}")
    print(f"    Tipo esperado: {item['expected']}")
    print("-" * 40)
    result = query_chain(regulatory_chain, item["question"])
    print(f"Respuesta:\n{result['answer']}")
    print("\nFuentes:")
    for src in result["sources"]:
        print(f"  • {src['source']}")