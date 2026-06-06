from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

def create_chunks(pages: list[dict], chunk_size: int=800, chunk_overlap: int=100) -> list[Document]:
    """
    Create chunks of text from a list of pages extracted from a PDF file.

    Args:
        pages (list[dict]): A list of dictionaries containing the page number and text content, coming from the pdf loader.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list[Document]: A list of Document objects containing the chunked text and metadata.
    """
    #splitter propio de langchain, divide el texto en fragmentos intentando conservar la estructura del texto
    splitter = RecursiveCharacterTextSplitter( 
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""] #Se prueban separadaores en orden de preferencia parrafos -> lineas -> oraciones -> palabras -> caracteres
        #Probar a quitar "." !!!!!!
    )
    
    documents = []
    
    for page in pages:
        # Document es un objeto de langchain que contiene el texto y la metadata asociada a ese texto
        text=page.pop("text")
        doc = Document(
            page_content=text,
            metadata=page
        )
        chunks = splitter.split_documents([doc]) # Se usa en vez de split_text para mantener el metadata
        documents.extend(chunks)
        
    logger.info(f"Chunking completed: {len(documents)} chunks made")
    return documents