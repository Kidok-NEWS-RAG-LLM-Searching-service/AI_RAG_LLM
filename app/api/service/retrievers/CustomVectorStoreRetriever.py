from langchain_core.vectorstores import VectorStoreRetriever


class CustomVectorStoreRetriever(VectorStoreRetriever):
    def _get_relevant_documents(self, query: str, **kwargs: object) -> object:
        """Customize how relevant documents are retrieved."""
        print("Custom retriever is being used.")
        # if self.search_type == "similarity":
        if self.search_type:
            score_threshold = self.search_kwargs.pop("score_threshold", 0)
            docs = self.vectorstore.similarity_search_with_score(
                query, **self.search_kwargs, **kwargs
            )
            if self.search_type == "similarity_score_threshold":
                docs_and_similarities = docs.copy()
                docs = [
                doc for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
        else:
            raise ValueError(f"Unknown search_type: {self.search_type}")
        return docs


