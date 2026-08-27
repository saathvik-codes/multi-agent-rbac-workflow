import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_GEN_MODEL: str = "llama3"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    DATABASE_PATH: str = "database.db"
    SIMILARITY_THRESHOLD: float = 0.60
    DEBUG: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __init__(self, **values):
        super().__init__(**values)
        # Attempt to get GEMINI_API_KEY from environment if not loaded from .env
        if not self.GEMINI_API_KEY:
            self.GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

settings = Settings()
