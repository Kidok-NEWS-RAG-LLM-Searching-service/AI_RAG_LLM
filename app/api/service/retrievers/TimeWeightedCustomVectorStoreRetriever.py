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

class TimeWeightedCustomVectorStoreRetriever(BaseRetriever):
    """Retriever that combines embedding similarity with recency in retrieving values."""

    vectorstore: VectorStore
    """The vectorstore to store documents and determine salience."""

    decay_rate: float = Field(default=0.01)
    """The exponential decay factor used as (1.0-decay_rate)**(hrs_passed)."""

    k: int = 4
    """The maximum number of documents to retrieve in a given call."""

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

    def _get_rescored_docs(self, docs_and_scores: List[Tuple[Document, float]]) -> List[Document]:
        """Rescore and sort the documents based on combined scores."""
        current_time = datetime.now()
        rescored_docs = [
            (doc, self._get_combined_score(score, doc, current_time))
            for doc, score in docs_and_scores
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        # return [(score, doc) for doc, score in rescored_docs[:self.k]]
        return [doc for doc, _ in rescored_docs[:self.k]]

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Retrieve and rescore documents based on query."""
        # Step 1: Retrieve documents using vectorstore with relevance scores
        # docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
        #     query,
        #     k=100,
        #     score_threshold = 0.67,
        # )

        docs_and_scores = CustomVectorStoreRetriever(
            vectorstore=self.vectorstore
        )._get_relevant_documents(
            query=query,
            k=50,
        )

        # Step 2: Rescore documents (combine vector relevance and time scores)
        rescored_docs = self._get_rescored_docs(docs_and_scores)
        return rescored_docs