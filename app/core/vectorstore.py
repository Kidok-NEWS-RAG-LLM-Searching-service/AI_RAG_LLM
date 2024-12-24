
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from typing import List, Any, Tuple, Optional
from langchain_core.vectorstores import VectorStoreRetriever
from app.api.service.retrievers.CustomVectorStoreRetriever import CustomVectorStoreRetriever

class CustomPineconeVectorStore(VectorStore):
    def __init__(self, base_store: PineconeVectorStore):
        self.base_store = base_store  # 기존 PineconeVectorStore 객체를 저장

    # 기존 PineconeVectorStore의 from_texts 호출
    @classmethod
    def from_texts(cls, texts: List[str], **kwargs: Any):
        return cls(base_store=PineconeVectorStore.from_texts(texts, **kwargs))

    # similarity_search 구현
    def similarity_search(self, query: str, **kwargs: Any) -> List[Document]:
        return self.base_store.similarity_search(query, **kwargs)

    def similarity_search_with_score(
            self, query: str, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with scores using custom logic."""
        print("Custom similarity_search_with_score is being used.")
        embedding = self.base_store._embedding.embed_query(query)
        # CustomVectorStoreRetriever에서 전달받은 self.search_kwargs와
        # 다른 곳에서 전달받은 kwargs를 합치기
        combined_kwargs = {}
        if hasattr(self, 'search_kwargs'):
            combined_kwargs.update(self.search_kwargs)
        combined_kwargs.update(kwargs)
        
        return self.similarity_search_by_vector_with_score(
            embedding, **combined_kwargs
        )

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        tags = kwargs.pop("tags", None) or [] + self._get_retriever_tags()
        return CustomVectorStoreRetriever(vectorstore=self, tags=tags, **kwargs)

    @staticmethod
    def _get_summary_docs(docs: List[Tuple[Document, float]]) -> list[Document]:
        print('get only summary page_content')
        summary_docs = docs.copy()
        for doc in summary_docs:
            doc[0].page_content = doc[0].page_content.split(" <Content>:")[0]
        return [doc for doc, score in summary_docs]
        # return summary_docs

    def similarity_search_by_vector_with_score(
            self,
            embedding: List[float],
            *,
            k: int = 4,
            filter: Optional[dict] = None,
            namespace: Optional[str] = None,
            setting: Optional[str] = None,
            search_type: Optional[str] = 'similarity',
            score_threshold: Optional[float] = 0,
            type: Optional[str] = None,
            **kwargs: Any
    ) -> list[Document] | list[tuple[Document, Any]]:
        """Return pinecone documents most similar to embedding, along with scores."""
        print('start similarity_search_by_vector_with_score')
        print('k: ', k)
        print('filter: ', filter)
        print('namespace: ', namespace)
        print('setting: ', setting)
        print('search_type: ', search_type)
        print('score_threshold: ', score_threshold)
        print('type: ', type)
        # print('search_kwargs: ', kwargs)
        # search_params = {
        #     'k': k,
        #     'filter': filter,
        #     'namespace': namespace,
        #     'setting': setting,
        #     'search_type': search_type,
        #     'score_threshold': score_threshold,
        #     'type': type
        # }
        
        # # search_kwargs의 값으로 업데이트
        # search_params.update(kwargs)
        
        # # 변수 할당
        # k = search_params['k']
        # filter = search_params['filter']
        # namespace = search_params['namespace']
        # setting = search_params['setting']
        # search_type = search_params['search_type']
        # score_threshold = search_params['score_threshold']
        # type = search_params['type']

        # print('search_params: ', search_params)
        # # `_text_key`를 안전하게 가져오기
        text_keys = getattr(self.base_store, "_text_key", [])
        if isinstance(text_keys, str):
            text_keys = [text_keys]  # 문자열인 경우 리스트로 변환
        if namespace is None:
            namespace = self._namespace
        docs = []
        if setting == 'no_query_embedding':
            print('setting is None')
            results = self._index.query(
            vector=[0]*4096,
            top_k=k,
            include_metadata=True,
            namespace=namespace,
            filter=filter,
        )
        else:
            results = self._index.query(
                vector=embedding,
                top_k=k,
                include_metadata=True,
                namespace=namespace,
                filter=filter,
            )
        # print(len(results["matches"]))
        for res in results["matches"]:
            metadata = res["metadata"]
            id = res.get("id")
            if any(k in metadata for k in text_keys):
                common_keys = [k for k in text_keys if k in metadata]
                text_parts = []

                for key in common_keys:
                    value = metadata.pop(key) if key == "content" else metadata.get(key, "")
                    # 리스트일 경우 문자열로 변환
                    if isinstance(value, list):
                        value = " ".join(value)
                    text_parts.append(key + ': ' + value + ' |')

                text = " ".join(text_parts)
                score = res["score"]
                docs.append(
                    (Document(id=id, page_content='id: ' + id[:-2] + ' |' + text, metadata=metadata), score)
                )
            else:
                print(
                    f"Found document with no `{self._text_key}` key. Skipping."
                )

        print('search_type: ', search_type)
        print('type: ', type)
        if setting == 'summary':
            # 요약 모델일때는 요약만 가져오기
            return self._get_summary_docs(docs)
        elif search_type == "similarity_score_threshold" and type is None:
            score_threshold_docs = []
            # print('start testtesttest')
            # print('score_threshold: ', score_threshold)
            docs_and_similarities = docs.copy()
            filtered_docs = [
                doc for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]   
            return filtered_docs
        elif search_type == "similarity_score_threshold" and type is not None:
            docs_and_similarities = docs.copy()
            filtered_docs = [
                (doc, similarity) for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            print('ready filtered_docs')
            return filtered_docs
        else:
            return docs

    # 나머지 메서드는 기본 PineconeVectorStore의 메서드를 호출
    def __getattr__(self, name):
        return getattr(self.base_store, name)

