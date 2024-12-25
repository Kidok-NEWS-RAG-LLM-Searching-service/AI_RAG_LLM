from datetime import datetime
from typing import List, Tuple, Any, Dict

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import Field

from app.api.service.retrievers.CustomVectorStoreRetriever import  CustomVectorStoreRetriever
from app.core.vectorstore import CustomPineconeVectorStore

from datetime import datetime

class TimeWeightedJounaralistFilteringVectorStoreRetriever(CustomVectorStoreRetriever):
    """Retriever that combines embedding similarity with recency in retrieving values."""

    vectorstore: CustomPineconeVectorStore = Field(default=None)
    decay_rate: float = Field(default=0.01)
    k: int = Field(default=4)
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
        merger = DynamicMerger()
        print('setting: ', self.search_kwargs.get('setting'))
        # print('search_kwargs: ', self.search_kwargs)
        print('filter_section: ', self.search_kwargs.get('filter')['section'])
        print('filter_init_year: ', self.search_kwargs.get('filter')['init_year'])
        rescored_docs: List[Document] = Field(default_factory=List[Document])
        for name in self.search_kwargs.get('name_list', []):
            print('name: ', name)
            docs_and_scores = CustomVectorStoreRetriever(
                vectorstore=self.vectorstore,
            )._get_relevant_documents(
                query=query,
                filter={
                    "journalist_name": {"$in": [name]},
                    "section": self.search_kwargs.get('filter')['section'],
                    "init_year": self.search_kwargs.get('filter')['init_year'],
                },
                k=self.k,
                setting=self.search_kwargs.get('setting')
            )
            # print('len(docs): ', len(docs_and_scores))

            # Step 2: Rescore documents (combine vector relevance and time scores)
            rescored_docs = self._get_rescored_docs(docs_and_scores)

            print('rescored_docs: ', len(rescored_docs))
            # page_content에서 요약했던 contextual 부분만 가져오기
            # summary_docs = self._get_summary_docs(rescored_docs)
            merger.add_list(rescored_docs)
        return merger.get_total_list()


# 기자들 관련 뉴스 합치는 class
class DynamicMerger:
    def __init__(self):
        # 저장된 리스트들을 관리할 구조
        self.lists = []
        # 결과를 저장할 total_list
        self.total_list = []

    def add_list(self, new_list: List[Any]):
        """
        새로운 리스트를 추가하고 즉시 total_list를 업데이트.
        Args:
            new_list: 새로 생성된 리스트
        """
        # 리스트가 비어있으면 pass
        if not new_list:
            return

        # 새 리스트를 저장
        self.lists.append(new_list)

        # 새로운 리스트를 포함하여 total_list를 업데이트
        self._update_total_list()

    def _update_total_list(self):
        """
        현재 저장된 모든 리스트를 interleave 방식으로 total_list에 병합.
        """
        # 초기화 (다시 쌓기)
        self.total_list = []

        # 가장 긴 리스트의 길이를 기준으로 반복
        max_length = max(len(lst) for lst in self.lists)

        for i in range(max_length):
            for lst in self.lists:
                if i < len(lst):  # 현재 리스트에 데이터가 있으면 추가
                    self.total_list.append(lst[i])

    def get_total_list(self) -> List[Any]:
        """
        현재까지 병합된 total_list 반환.
        """
        return self.total_list
