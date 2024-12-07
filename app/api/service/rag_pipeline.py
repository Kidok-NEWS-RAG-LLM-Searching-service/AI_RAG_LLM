import os
from datetime import datetime
from typing import Any, List

import pandas as pd
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app.api.service.encoders.encoders import sparse_encoder
from app.api.service.managers.stop_words_manager import StopwordsManager
from app.api.service.retrievers.PineconeKiwiHybridRetriever import PineconeKiwiHybridRetriever
from app.core.llm import AIModelManager
from app.core.pinecone_index_initializer import PineconeIndexInitializer

from typing import AsyncGenerator


current_dir = os.path.dirname(os.path.abspath(__file__))


class RagPipeline:
    stop_words_manager = StopwordsManager()
    ai_model_manager = AIModelManager()

    llm = ai_model_manager.llm
    embeddings = ai_model_manager.embeddings

    sparse_encoder_path = os.path.join("./news_rag_llm/yong_contextual_sparse_encoder.pkl")

    global_source_set = set()

    if not os.path.exists(sparse_encoder_path):
        print(f"{sparse_encoder_path} not found. Creating sparse encoder...")
        contextual_chunks_df = pd.read_csv("./app/assets/contextual_chunks_df.csv")
        sparse_encoder_value = sparse_encoder.create_sparse_encoder(
            stop_words_manager.fetch_stopwords(), mode="kiwi"
        )
        saved_path = sparse_encoder.fit(
            bm25_encoder=sparse_encoder_value,
            contents=contextual_chunks_df.contexts.tolist(),
            save_path=sparse_encoder_path
        )
        print(f"Sparse encoder saved at: {saved_path}")

    pinecone_index_initializer = PineconeIndexInitializer(
        sparse_encoder_path="./news_rag_llm/yong_contextual_sparse_encoder.pkl",
        stopwords=stop_words_manager.fetch_stopwords(),  # 불용어 사전
        tokenizer="kiwi",
        embeddings=embeddings,
        top_k=5,
        alpha=0.7,
    )

    init_data = pinecone_index_initializer.get_pinecone_init_data()

    pinecone_retriever = PineconeKiwiHybridRetriever(
        embeddings=init_data["embeddings"],
        sparse_encoder=init_data["sparse_encoder"],
        index=init_data["index"],
        top_k=init_data["top_k"],
        alpha=init_data["alpha"],
        namespace=init_data["namespace"]
    )

    prompt = ChatPromptTemplate.from_template(AIModelManager.get_custom_prompt_template())
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(pinecone_retriever, question_answer_chain)

    def query(self, query: str):
        rag_result = self.rag_chain.invoke(
            {
                "input": query,
                "current_time": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
                "MAX_TOKENS": self.ai_model_manager.DEFAULT_MAX_TOKEN
            }
        )

        source_set = set()
        for document in rag_result.get("context"):
            meta = document.metadata
            source_set.add(
                f"source: {meta['source']}, title: {meta['title']}  {meta['mod_date']} {meta['mod_timestamp']}"
            )
        response = {
            "rag_result": rag_result.get("answer", "No answer generated"),
            "sources": source_set
        }
        return response


    def get_documents(self, query: str, k = 5) -> List[Document]:
        docs = self.pinecone_retriever.invoke(query, k=k)
        return docs

    @staticmethod
    def get_source(docs: List[Document]) -> List[dict]:
        source_list = []
        for document in docs:
            meta = document.metadata
            new_item = {
                    "source": meta["source"],
                    "title": meta["title"],
                    "date": f"{meta['mod_date']} {meta['mod_timestamp']}",
                    "image_url": meta["images_url"],
                }
            if new_item not in source_list:
                source_list.append(new_item)
        return source_list

    async def stream_query(self, query: str, docs: List[Document]) -> AsyncGenerator[str, None]:
        try:
            async for event in self.question_answer_chain.astream(
                    {
                        "input": query,
                        'context': docs,
                        "current_time": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
                        "MAX_TOKENS": self.ai_model_manager.DEFAULT_MAX_TOKEN
                    }
            ):
                yield event  # 스트리밍 데이터 전송

        except Exception as e:
            yield f"Error: {str(e)}\n"


rag_pipeline = RagPipeline()
