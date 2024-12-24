from datetime import datetime
from typing import List, Tuple
from pydantic import Field
from langchain_core.documents import Document

from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
)

from app.api.service.retrievers.CustomVectorStoreRetriever import CustomVectorStoreRetriever

# class TimeWeightedCustomVectorStoreRetriever(BaseRetriever):
#     """Retriever that combines embedding similarity with recency in retrieving values."""

#     vectorstore: VectorStore
#     """The vectorstore to store documents and determine salience."""

#     decay_rate: float = Field(default=0.01)
#     """The exponential decay factor used as (1.0-decay_rate)**(hrs_passed)."""

#     k: int = 4
#     """The maximum number of documents to retrieve in a given call."""


#     search_kwargs: dict = Field(default=dict)
#     """The search kwargs to pass to the vectorstore."""

#     def _document_get_datetime(self, document: Document, field: str) -> datetime:
#         """Convert metadata field to datetime."""
#         if field in document.metadata:
#             return datetime.strptime(document.metadata[field], "%Y-%m-%d")
#         return datetime.now()

#     def _get_combined_score(
#             self,
#             vector_relevance: float,
#             document: Document,
#             current_time: datetime,
#     ) -> float:
#         """Calculate combined score (vector relevance + time score)."""
#         init_date = self._document_get_datetime(document, "init_date")
#         hours_passed = (current_time - init_date).total_seconds() / 3600
#         time_score = (1.0 - self.decay_rate) ** hours_passed
#         return vector_relevance + time_score

#     def _get_rescored_docs(self, docs_and_scores: List[Tuple[Document, float]]) -> List[Document]:
#         """Rescore and sort the documents based on combined scores."""
#         current_time = datetime.now()
#         rescored_docs = [
#             (doc, self._get_combined_score(score, doc, current_time))
#             for doc, score in docs_and_scores
#         ]
#         rescored_docs.sort(key=lambda x: x[1], reverse=True)
#         # return [(score, doc) for doc, score in rescored_docs[:self.k]]
#         return [doc for doc, _ in rescored_docs[:self.k]]

#     def _get_relevant_documents(
#             self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
#     ) -> List[Document]:
#         """Retrieve and rescore documents based on query."""

#         docs_and_scores = CustomVectorStoreRetriever(
#             vectorstore=self.vectorstore,
#         )._get_relevant_documents(
#             query=query,
#             k=40, # 40개 중에서 score에 따라 self.k개를 추출
#             search_type=self.search_type,
#             **self.search_kwargs
#         )

#         # Step 2: Rescore documents (combine vector relevance and time scores)
#         rescored_docs = self._get_rescored_docs(docs_and_scores)
#         return rescored_docs


from app.api.service.retrievers.CustomVectorStoreRetriever import  CustomVectorStoreRetriever
from app.core.vectorstore import CustomPineconeVectorStore

class TimeWeightedCustomVectorStoreRetriever(CustomVectorStoreRetriever):
    """Retriever that combines embedding similarity with recency in retrieving values."""

    vectorstore: CustomPineconeVectorStore = Field(default=None)
    decay_rate: float = Field(default=0.01)
    # k: int = Field(default=4)
    # search_type: str = Field(default='similarity_score_threshold')
    search_kwargs: dict = Field(default=dict)
    
    def __init__(self, vectorstore: CustomPineconeVectorStore, **kwargs):
        if not isinstance(vectorstore, CustomPineconeVectorStore):
            raise TypeError("vectorstore must be an instance of CustomPineconeVectorStore")
        super().__init__(vectorstore=vectorstore, **kwargs)

    def _document_get_datetime(self, document: Document, field: str) -> datetime:
        """Convert metadata field to datetime."""
        if field in document.metadata:
            return datetime.strptime(document.metadata[field], "%Y-%m-%d")
        return datetime.now()

    def _get_combined_score(
            self,
            vector_relevance: float,
            document: Document,
            current_time: datetime,
    ) -> float:
        """Calculate combined score (vector relevance + time score)."""
        init_date = self._document_get_datetime(document, "init_date")
        hours_passed = (current_time - init_date).total_seconds() / 3600
        time_score = (1.0 - self.decay_rate) ** hours_passed
        return vector_relevance + time_score

    def _get_rescored_docs(self, docs_and_scores: List[Tuple[Document, float]], k: int) -> List[Document]:
        """Rescore and sort the documents based on combined scores."""
        current_time = datetime.now()
        print('start rescored_docs_in code')
        rescored_docs = [
            (doc, self._get_combined_score(score, doc, current_time))
            for doc, score in docs_and_scores
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        # return [(score, doc) for doc, score in rescored_docs[:self.k]]
        return [doc for doc, _ in rescored_docs[:k]]

    def _get_summary_docs(self, rescored_docs: List[Document]) -> List[Document]:
        summary_docs = rescored_docs.copy()
        for doc in summary_docs:
            temp = doc.page_content.split(" <Content>:")

            # journalist_name 추출 및 추가
            if "journalist_name:" in temp[1]:
                try:
                    # 정규식을 사용하여 journalist_name 값을 추출
                    import re
                    match = re.search(r"journalist_name:\s*([^|]+)", temp[1])
                    if match:
                        journalist_name = match.group(1).strip()
                        # 메타데이터에 추가
                        doc.page_content = journalist_name
                except Exception as e:
                    print(f"Error extracting journalist_name: {e}")

            doc.page_content += temp[0]
        return summary_docs

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Document]:
        """Retrieve and rescore documents based on query."""
        # 사용 예제
        # print('search_type: ', self.search_type)
        rescored_docs: List[Document] = Field(default_factory=List[Document])
        
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query=query,
            k=40,
            **self.search_kwargs,
        )
        # docs_and_scores = CustomVectorStoreRetriever(
        #     vectorstore=self.vectorstore,
        #     # search_type=self.search_type,
        # )._get_relevant_documents(
        #     query=query,
        #     k=40, # 40개 중에서 score에 따라 self.k개를 추출
        #     **self.search_kwargs,
        # )
        print('len(docs): ', len(docs_and_scores))

        # Step 2: Rescore documents (combine vector relevance and time scores)
        print('start rescored_docs')
        rescored_docs = self._get_rescored_docs(docs_and_scores, k=20)

        # page_content에서 요약했던 contextual 부분만 가져오기
        # summary_docs = self._get_summary_docs(rescored_docs)

        return rescored_docs