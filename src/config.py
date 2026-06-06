from dataclasses import dataclass

@dataclass
class Config:
    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    # Retrieval
    TOP_K_FINANCIAL: int = 4
    TOP_K_REGULATORY: int = 6    # más chunks para normativa — respuestas más completas

    # LLM
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    TEMPERATURE: float = 0.0

    # Paths de colecciones
    CHROMA_FINANCIAL_DIR: str = "chroma_db/financial"
    CHROMA_REGULATORY_DIR: str = "chroma_db/regulatory"

    # Carpetas de datos
    DATA_FINANCIAL_DIR: str = "data/financial"
    DATA_REGULATORY_DIR: str = "data/regulatory"
    
config = Config()