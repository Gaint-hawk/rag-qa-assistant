from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Empty = in-memory vector store. A PostgreSQL URL with the pgvector extension
    # enables persistent storage: postgresql+psycopg://user:pass@host:5432/rag
    database_url: str = ""

    # "fastembed" runs BAAI/bge-small-en-v1.5 locally (no API key); "hash" is a
    # dependency-free embedder for tests and offline demos.
    embedder: str = "fastembed"

    claude_model: str = "claude-opus-5-5"
    chunk_words: int = 180
    chunk_overlap_words: int = 40
    top_k: int = 5
    min_score: float = 0.2
    cors_origins: list[str] = ["http://localhost:5173"]


settings = Settings()
