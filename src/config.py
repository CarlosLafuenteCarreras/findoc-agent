from dataclasses import dataclass

@dataclass
class Config:
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    TOP_K_CHUNKS: int = 4
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    TEMPERATURE: float = 0.0
    CHROMA_PERSIST_DIR: str = "chroma_db"

config = Config()