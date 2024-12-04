from pathlib import Path

from pydantic.v1 import BaseSettings


class Settings(BaseSettings):

    # Static Variable
    API_V1_STR: str = "/api/v1"
    CLIENT_CORS_ORIGIN: list[str] = ["http://localhost:3000"]
    CORS_ORIGIN: list[str] = ["http://localhost:8000"]

    # Dynamic Variable
    env: str
    debug: bool = True
    # spring_api_server_origin: str
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str
    # pinecone_environment: str
    upstage_api_key: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs, _env_file=Path(f".env.{kwargs.get('env', 'dev')}"))

    @property
    def all_cors_origin(self) -> list[str]:
        return self.CLIENT_CORS_ORIGIN + self.CORS_ORIGIN

    @property
    def get_openai_api_key(self) -> str:
        return self.openai_api_key

    @property
    def get_pinecone_api_key(self) -> str:
        return self.pinecone_api_key

    @property
    def get_pinecone_index_name(self) -> str:
        return self.pinecone_index_name

    @property
    def get_pinecone_namespace(self) -> str:
        return self.pinecone_namespace

    @property
    def get_upstage_api_key(self) -> str:
        return self.upstage_api_key


settings = Settings()
