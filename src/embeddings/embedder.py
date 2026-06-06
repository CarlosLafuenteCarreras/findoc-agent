from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

# Separamos en 2 funciones, 1 para crear el vectorstore a partir de los documentos y otra para cargarlo desde disco, asi evitamos crear un nuevo vectorstore cada vez que queramos hacer una consulta al agente, lo que seria ineficiente y costoso en terminos de tiempo y recursos.

def create_financial_vectorstore(documents: list[Document], persist_directory: str = "chroma_db/financial") -> Chroma:
    """
    Create a Chroma vector store from a list of Document objects.

    Args:
        documents (list[Document]): A list of Document objects containing the text and metadata.
        persist_directory (str): The directory to persist the vector store.

    Returns:
        Chroma: A Chroma vector store containing the embeddings of the documents.
    """
    # Crear el vectorstore usando Chroma y OpenAIEmbeddings
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    vectorstore = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory)

    # guardamos en disco
    vectorstore.persist()
    
    print(f"Vector store of {len(documents)} chunks created and persisted in directory: {persist_directory}.")
    
    return vectorstore



def create_regulatory_vectorstore(documents: list[Document], persist_directory: str = "chroma_db/regulatory") -> Chroma:
    """
    Create a Chroma vector store from a list of Document objects.

    Args:
        documents (list[Document]): A list of Document objects containing the text and metadata.
        persist_directory (str): The directory to persist the vector store.

    Returns:
        Chroma: A Chroma vector store containing the embeddings of the documents.
    """
    # Crear el vectorstore usando Chroma y OpenAIEmbeddings
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    vectorstore = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory)

    # guardamos en disco
    vectorstore.persist()
    
    print(f"Vector store of {len(documents)} chunks created and persisted in directory: {persist_directory}.")
    
    return vectorstore




def load_financial_vectorstore(persist_directory: str = "chroma_db/financial") -> Chroma:
    """
    Load a Chroma vector store from a specified directory.

    Args:
        persist_directory (str): The directory where the vector store is persisted.

    Returns:
        Chroma: A Chroma vector store loaded from the specified directory.
    """
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    return Chroma(persist_directory=persist_directory, embedding_function=embedding)



def load_regulatory_vectorstore(persist_directory: str = "chroma_db/regulatory") -> Chroma:
    """
    Load a Chroma vector store from a specified directory.

    Args:
        persist_directory (str): The directory where the vector store is persisted.

    Returns:
        Chroma: A Chroma vector store loaded from the specified directory.
    """
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    return Chroma(persist_directory=persist_directory, embedding_function=embedding)