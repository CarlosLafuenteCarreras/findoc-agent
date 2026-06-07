from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)

from src.embeddings.embedder import ingest_financial_folder, ingest_regulatory_folder

print("===============================")
print("INDEXING FINANCIAL DOCUMENTS")
print("===============================")
financial_vs = ingest_financial_folder()


print("===============================")
print("INDEXING REGULATORY DOCUMENTS")
print("===============================")
regulatory_vs = ingest_regulatory_folder()


print("===============================")
print("TRYING RETRIEVAL")
print("===============================")

query_fin = "beneficio neto santander"

results_fin = financial_vs.similarity_search(query_fin, k=3)
print(f"Financial query:{query_fin}")

for i, doc in enumerate(results_fin, 1):
    print("-" * 40)
    print(f"Result {i}:")
    print(f"{doc.page_content}\n From: {doc.metadata.get('source')} of {doc.metadata.get('company')},{doc.metadata.get('year')} - Pag. {doc.metadata.get('page')}") 
    print("-" * 40)
    
query_reg = "requisitos en el modelado de PD en IRB"

results_reg = regulatory_vs.similarity_search(query_reg, k=3)
print(f"Regulatory query:{query_reg}")

for i, doc in enumerate(results_reg, 1):
    print("-" * 40)
    print(f"Result {i}:")
    print(f"{doc.page_content}\n From: {doc.metadata.get('source')} of {doc.metadata.get('regulation')},{doc.metadata.get('year')} - Pag. {doc.metadata.get('page')}") 
    print("-" * 40)