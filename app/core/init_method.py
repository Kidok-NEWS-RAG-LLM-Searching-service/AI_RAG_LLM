from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from app.core.config import settings
from app.core.llm import AIModelManager

class InitVectorStore:

    ai_model_manager = AIModelManager()

    def init_pinecone_vectorstore(self):
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        text_field = "content"

        return PineconeVectorStore(
            index,
            embedding=self.ai_model_manager.embeddings,
            text_key=text_field,
            namespace=''
        )

    def init_customize_vectorstore(self):
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        text_field = ["content", "journalist_name"]

        return PineconeVectorStore(
            index=index,
            embedding=self.ai_model_manager.embeddings,
            text_key=text_field,
            namespace=''
        )
