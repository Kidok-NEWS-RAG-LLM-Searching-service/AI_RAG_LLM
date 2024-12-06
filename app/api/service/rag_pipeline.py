from tensorflow.python.checkpoint.checkpoint_management import update_checkpoint_state

from AI_RAG_LLM.app.core.pinecone import PineconeConfig, PineconeKiwiHybridRetriever
from AI_RAG_LLM.app.core.upstage import UpstageConfig
from AI_RAG_LLM.app.core.openai import OpenaiConfig
from prompt import qa_prompt

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from datetime import datetime


class RagPipeline():
    pinecone_config = PineconeConfig()
    upstage_config = UpstageConfig()
    openai_config = OpenaiConfig()

    def __init__(self):
        # Pinecone
        self.pinecone_params = self.pinecone_config.init_pinecone_index(
            index_name=self.pinecone_config.index_name,
            namespace=self.pinecone_config.namespace,
            api_key=self.pinecone_config.pinecone_api_key,
            sparse_encoder_path=self.pinecone_config.SPARSE_ENCODER_PTAH,
            stopwords=self.pinecone_config.stopwords(),
            tokenizer='kiwi',
            embeddings=self.upstage_config.contents_embedding,
            top_k=5,
            alpha=0.5,
        )
        self.hybrid_retriever = self.__init_hybrid_retriever__()

        # LLM
        self.rag_chain = self.__init_qa_chin__()

    def __init_hybrid_retriever__(self):
        return PineconeKiwiHybridRetriever(**self.pinecone_params)

    def __init_qa_chin__(self):
        question_answer_chain = create_stuff_documents_chain(self.openai_config.llm, qa_prompt)
        rag_chain = create_retrieval_chain(self.hybrid_retriever, question_answer_chain)
        return rag_chain

    # Answer
    def generate_llm_result(self, query: str)-> dict:
        result = self.rag_chain.invoke(
            {
                "input": query,
                "current_time": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
                "MAX_TOKENS": self.openai_config.MAX_TOKENS
            }
        )
        return result
