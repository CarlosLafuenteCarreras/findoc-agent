from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import logging
from pathlib import Path
from src.config import config

# Separamos en 2 funciones, 1 para crear el vectorstore a partir de los documentos y otra para cargarlo desde disco, asi evitamos crear un nuevo vectorstore cada vez que queramos hacer una consulta al agente, lo que seria ineficiente y costoso en terminos de tiempo y recursos.


logger = logging.getLogger(__name__)


def _get_embeddings() -> OpenAIEmbeddings:
    """
    Get the OpenAIEmbeddings object with the specified model and API key.

    Returns:
        OpenAIEmbeddings: An instance of the OpenAIEmbeddings class.
    """
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )


def create_financial_vectorstore(documents: list[Document]) -> Chroma:
    """
    Create a Chroma vector store from a list of Document objects.

    Args:
        documents (list[Document]): A list of Document objects containing the text and metadata.

    Returns:
        Chroma: A Chroma vector store containing the embeddings of the documents.
    """
    # Crear el vectorstore usando Chroma y OpenAIEmbeddings
    embedding = _get_embeddings()
    
    vectorstore = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=config.CHROMA_FINANCIAL_DIR)

    
    logger.info(f"Financial vector store created: {len(documents)} chunks")
    
    return vectorstore



def create_regulatory_vectorstore(documents: list[Document]) -> Chroma:
    """
    Create a Chroma vector store from a list of Document objects.

    Args:
        documents (list[Document]): A list of Document objects containing the text and metadata.

    Returns:
        Chroma: A Chroma vector store containing the embeddings of the documents.
    """
    # Crear el vectorstore usando Chroma y OpenAIEmbeddings
    embedding = _get_embeddings()
    
    vectorstore = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=config.CHROMA_REGULATORY_DIR)

    
    logger.info(f"Regulatory vector store created: {len(documents)} chunks")
    
    return vectorstore




def load_financial_vectorstore() -> Chroma:
    """
    Load a Chroma vector store from a specified directory.

    Returns:
        Chroma: A Chroma vector store loaded from the specified directory.
    """
    embedding = _get_embeddings()
    
    return Chroma(persist_directory=config.CHROMA_FINANCIAL_DIR, embedding_function=_get_embeddings())


def load_regulatory_vectorstore() -> Chroma:
    """
    Load a Chroma vector store from a specified directory.

    Returns:
        Chroma: A Chroma vector store loaded from the specified directory.
    """
    embedding = _get_embeddings()
    
    return Chroma(persist_directory=config.CHROMA_REGULATORY_DIR, embedding_function=_get_embeddings())




def add_documents_to_financial(new_documents: list[Document]) -> Chroma:
    """
    Add new documents to the existing financial vector store.

    Args:
        new_documents (list[Document]): A list of Document objects to be added to the vector store.
    
    Returns:
        Chroma: The updated Chroma vector store with the new documents added.
    """
    vectorstore = load_financial_vectorstore()
    vectorstore.add_documents(new_documents)
    logger.info(f"Added {len(new_documents)} new documents to the financial vector store.")
    
    
def add_documents_to_regulatory(new_documents: list[Document]) -> Chroma:
    """
    Add new documents to the existing regulatory vector store.

    Args:
        new_documents (list[Document]): A list of Document objects to be added to the vector store.
    
    Returns:
        Chroma: The updated Chroma vector store with the new documents added.
    """
    vectorstore = load_regulatory_vectorstore()
    vectorstore.add_documents(new_documents)
    logger.info(f"Added {len(new_documents)} new documents to the regulatory vector store.")
    

def get_indexed_sources(vectorstore:Chroma) -> list[str]:
    """
    Get a list of already indexed sources from the vector store metadata.
    """
    try:
        collection = vectorstore._collection.get()
        sources = set()
        for metadata in collection['metadatas']:
            if metadata.get("source"):
                sources.add(metadata["source"])
        return list(sources)
        
    except Exception:
        return []
   
   
def ingest_financial_folder() -> Chroma:
    """
    Ingest all PDF files from the financial data folder, create chunks with metadata, and add them to the financial vector store.
    """
    from src.ingestion.pdf_loader import load_pdf
    from src.ingestion.chunker import create_chunks
    from src.ingestion.metadata_parser import parse_financial_metadata

    folder = Path(config.DATA_FINANCIAL_DIR)
    all_chunks = []

    # Determina si ya existe la colección para cargarla o crearla
    financial_dir = Path(config.CHROMA_FINANCIAL_DIR)
    if financial_dir.exists():
        vectorstore = load_financial_vectorstore()
        already_indexed = get_indexed_sources(vectorstore)
    else:
        vectorstore = None
        already_indexed = []

    for pdf_file in folder.glob("*.pdf"):
        if pdf_file.name in already_indexed:
            logger.info(f"Ya indexado, saltando: {pdf_file.name}")
            continue

        logger.info(f"Procesando: {pdf_file.name}")
        pages = load_pdf(pdf_file)

        # Inyecta la metadata parseada en cada página
        metadata = parse_financial_metadata(pdf_file.name)
        for page in pages:
            page.update(metadata)

        chunks = create_chunks(pages)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.info("No hay documentos nuevos que indexar en financial")
        return vectorstore or load_financial_vectorstore()

    if vectorstore is None:
        return create_financial_vectorstore(all_chunks)
    else:
        return add_documents_to_financial(all_chunks)


def ingest_regulatory_folder() -> Chroma:
    """
    Ingest all PDF files from the regulatory data folder, create chunks with metadata, and add them to the regulatory vector store.
    """
    from src.ingestion.pdf_loader import load_pdf
    from src.ingestion.chunker import create_chunks
    from src.ingestion.metadata_parser import parse_regulatory_metadata

    folder = Path(config.DATA_REGULATORY_DIR)
    all_chunks = []

    regulatory_dir = Path(config.CHROMA_REGULATORY_DIR)
    if regulatory_dir.exists():
        vectorstore = load_regulatory_vectorstore()
        already_indexed = get_indexed_sources(vectorstore)
    else:
        vectorstore = None
        already_indexed = []

    for pdf_file in folder.glob("*.pdf"):
        if pdf_file.name in already_indexed:
            logger.info(f"Ya indexado, saltando: {pdf_file.name}")
            continue

        logger.info(f"Procesando: {pdf_file.name}")
        pages = load_pdf(pdf_file)

        # enriquece el texto con metadatos
        metadata = parse_regulatory_metadata(pdf_file.name)
        for page in pages:
            page.update(metadata)

        chunks = create_chunks(pages)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.info("No hay documentos nuevos que indexar en regulatory")
        return vectorstore or load_regulatory_vectorstore()

    if vectorstore is None:
        return create_regulatory_vectorstore(all_chunks)
    else:
        return add_documents_to_regulatory(all_chunks)
