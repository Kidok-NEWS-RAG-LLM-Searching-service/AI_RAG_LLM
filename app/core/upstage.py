from AI_RAG_LLM.app.core.config import settings
from langchain_upstage import UpstageEmbeddings

class UpstageConfig:
    CONTENTS_EMBEDDING_MODEL = "solar-embedding-1-large-passage"
    QUERY_EMBEDDING_MODEL = "solar-embedding-1-large-query"
    upstage_api_key = settings.get_upstage_api_key()

    def __int__(self):
        self.contents_embedding = UpstageEmbeddings(
            model=self.CONTENTS_EMBEDDING_MODEL,
            api_key=self.upstage_api_key
            )
        self.query_embedding = UpstageEmbeddings(
            model=self.QUERY_EMBEDDING_MODEL,
            api_key=self.upstage_api_key
            )





