import fitz
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Load a PDF file and extract its text content.

    Args:
        pdf_path (str | Path): The path to the PDF file.
        
    Returns:
        list[dict]: A list of dictionaries containing the page number and text content.
    """
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    doc = fitz.open(str(pdf_path))
    pages = []
    
    for page_num in range(len(doc)):
        
        # Extraigo el texto de la pagina
        page = doc[page_num]
        text = page.get_text()
        
        # Limpiamos el texto eliminando lineas vacias 
        text = "\n".join(
            line for line in text.split("\n")
            if line.strip() # True si la linea no esta vacia
        )
        
        if text.strip():
            pages.append({
                "text": text,
                "page": page_num+1,
                "source": pdf_path.name
            })
            
    doc.close()
    logger.info(f"PDF cargado: {pdf_path.name} — {len(pages)} páginas con texto")
    return pages
