from pathlib import Path

from pydantic.v1 import BaseSettings


class Settings(BaseSettings):

    # Static Variable
    API_V1_STR: str = "/api/v1"
    CLIENT_CORS_ORIGIN: list[str] = ["https://kidokai.com", "http://localhost:5173"]
    CORS_ORIGIN: list[str] = ["http://api.kidokai.com"]

    # Dynamic Variable
    env: str
    debug: bool = True
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str
    upstage_api_key: str

    mongo_db_name: str
    mongo_db_url: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs, _env_file=Path(f".env.{kwargs.get('env', 'dev')}"))

    @property
    def all_cors_origin(self) -> list[str]:
        return self.CLIENT_CORS_ORIGIN + self.CORS_ORIGIN


settings = Settings()
