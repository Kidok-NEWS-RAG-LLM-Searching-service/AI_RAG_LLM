from langchain_core.vectorstores import VectorStoreRetriever
from typing import List, Tuple
from langchain_core.documents import Document



class CustomVectorStoreRetriever(VectorStoreRetriever):
    def _get_relevant_documents(self, query: str, **kwargs: object) -> List[Tuple[Document, float]]:
        """Customize how relevant documents are retrieved."""
        # print("Custom retriever is being used.")
        # search_type = kwargs.pop("search_type", "similarity")
        # print('search_type: ', self.search_type)
        # if search_type:
        # score_threshold = self.search_kwargs.pop("score_threshold", 0)
        # print('kwargs: ', self.search_kwargs)
        docs = self.vectorstore.similarity_search_with_score(
            query, **self.search_kwargs, **kwargs
        )
        return docs


