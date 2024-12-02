from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from app.core.pinecone import PineconeConfig
from upstaage import UpstageConfig
from openai import OpenaiConfig
from prompt import qa_prompt
from datetime import datetime


class RagPipeline():
    pinecone_config = PineconeConfig()
    upstage_config = UpstageConfig()
    openai_config = OpenaiConfig()

    def __init__(self):
        # Pinecone
        # self.pinecone_params = {
        #     index_name=self.pinecone_config.index_name,  # Pinecone 인덱스 이름
        #     namespace=self.pinecone_config.namespace,  # Pinecone Namespace
        #     api_key= self.pinecone_config.pinecone_api_key,  # Pinecone API Key
        #     sparse_encoder_path=self.pinecone_config.sparse_encoder_path,  # Sparse Encoder 저장경로(save_path)
        #     stopwords=self.pinecone_config.stopwords(),  # 불용어 사전
        #     tokenizer=self.pinecone_config.tokenizer,
        #     embeddings=self.upstage_config.embeddings,  # Dense Embedder
        #     top_k=self.pinecone_config.top_k,  # Top-K 문서 반환 개수
        #     alpha=self.pinecone_config.alpha
        #     }
        self.hybrid_retriever = self.__init_hybrid_retriever__()

        # LLM
        self.rag_chain = self.__init_qa_chin__()

        # Answer
        self.result = self.get_llm_result(query)
        self.answer = self.get_answer(self.result)





    def __init_hybrid_retriever__(self):
        retreiver = self.pinecone_config.PineconeKiwiHybridRetriever()
        return retreiver

    def __init_qa_chin__(self):
        question_answer_chain = create_stuff_documents_chain(self.openai_config.llm, qa_prompt)
        rag_chain = create_retrieval_chain(self.hybrid_retriever, question_answer_chain)
        return rag_chain

    def get_llm_result(self, query: str)-> dict:
        result = self.rag_chain.invoke(
            {
                "input": query,
                "current_time": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
                "MAX_TOKENS": self.openai_config.MAX_TOKENS
            }
        )
        return result

    def get_answer(self, result: dict)-> str:
        return result['answer']

    def get_metadata(self, result: dict)-> list:
        return result['context']