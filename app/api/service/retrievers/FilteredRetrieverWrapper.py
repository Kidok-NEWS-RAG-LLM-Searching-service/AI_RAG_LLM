from langchain_core.retrievers import BaseRetriever
from typing import Dict, Any, List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document


class FilteredRetrieverWrapper(BaseRetriever):
    def __init__(self, base_retriever: BaseRetriever, filters: Dict[str, Any]):
        super().__init__()
        self.base_retriever = base_retriever
        self.filters = filters
        
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs
    ) -> List[Document]:
        # 기존 search_kwargs와 merge하여 필터 추가
        search_kwargs = kwargs.get("search_kwargs", {})
        search_kwargs["filter"] = self.filters

        # base_retriever에서 `_get_relevant_documents` 호출
        return self.base_retriever._get_relevant_documents(
            query, run_manager=run_manager, search_kwargs=search_kwargs
        )

    def invoke(self, query: str, **kwargs) -> Any:
        # 기존 search_kwargs와 merge하여 필터 추가
        search_kwargs = kwargs.get("search_kwargs", {})
        search_kwargs["filter"] = self.filters
        return self.base_retriever.invoke(query, search_kwargs=search_kwargs)

    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)