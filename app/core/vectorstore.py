from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from typing import List, Any, Tuple, Optional
from langchain_core.vectorstores import VectorStoreRetriever
from app.api.service.retrievers.CustomVectorStoreRetriever import CustomVectorStoreRetriever

from functools import wraps
import time

from concurrent.futures import ThreadPoolExecutor
import functools
import asyncio
import multiprocessing

class CustomPineconeVectorStore(VectorStore):
    def __init__(self, base_store: PineconeVectorStore):
        self.base_store = base_store  # 기존 PineconeVectorStore 객체를 저장
        cpu_count = multiprocessing.cpu_count()
        print('CustomPineconeVectorStore __init__ - cpu_count*2: ', cpu_count*2)
        self.thread_pool = ThreadPoolExecutor(max_workers=cpu_count * 2)

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
        # print("Custom similarity_search_with_score is being used.")
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

    def timer(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            print(f" | {func.__name__} 실행 시간: {end_time - start_time:.2f}초 | ")
            return result
        return wrapper
    
    def _get_summary_docs(self,docs: List[Tuple[Document, float]]) -> list[Document]:
        print('get only summary page_content')
        summary_docs = docs.copy()
        for doc in summary_docs:
            doc[0].page_content = doc[0].page_content.split(" <Content>:")[0]
        # print('summary_docs: ', len(summary_docs))
        return [doc for doc, _ in summary_docs]
        # return summary_docs

    def similarity_search_by_vector_with_score(
            self,
            embedding: List[float],
            *,
            k: int = 4,
            filter: Optional[dict] = None,
            namespace: Optional[str] = None,
            setting: Optional[str] = "None",
            search_type: Optional[str] = 'similarity',
            score_threshold: Optional[float] = 0,
            **kwargs: Any
    ) -> list[Document] | list[tuple[Document, float]]:
        """Return pinecone documents most similar to embedding, along with scores."""
        # print('start similarity_search_by_vector_with_score')
        # print('k: ', k)
        # print('filter: ', filter)
        # print('namespace: ', namespace)
        # print('setting: ', setting)
        # print('search_type: ', search_type)
        # print('score_threshold: ', score_threshold)
        # print('search_kwargs: ', kwargs)


        if namespace is None:
            namespace = self._namespace

        if 'no_query_embedding' in setting:
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
        
        docs = []
        # # `_text_key`를 안전하게 가져오기
        text_keys = getattr(self.base_store, "_text_key", [])
        if isinstance(text_keys, str):
            text_keys = [text_keys]  # 문자열인 경우 리스트로 변환
            
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

        # print('search_type: ', search_type)
        # print('setting: ', setting)
        if 'summary' in setting:
            # 요약 모델일때는 요약만 가져오기
            return self._get_summary_docs(docs)
        elif setting == 'time_weighted':
            # print('ready filtered_docs')
            return [
                (doc, similarity) for doc, similarity in docs
                if similarity >= score_threshold
            ]
        elif search_type == "similarity_score_threshold":
            score_threshold_docs = []
            # print('start testtesttest')
            # print('score_threshold: ', score_threshold)
            return [
                doc for doc, similarity in docs
                if similarity >= score_threshold
            ]   
        else:
            return docs
        
    # 비동기 메서드 추가
    async def asimilarity_search_by_vector_with_score(
            self,
            embedding: List[float],
            *,
            k: int = 4,
            filter: Optional[dict] = None,
            namespace: Optional[str] = None,
            setting: Optional[str] = "None",
            search_type: Optional[str] = 'similarity',
            score_threshold: Optional[float] = 0,
            **kwargs: Any
    ) -> list[Document] | list[tuple[Document, float]]:
        """비동기 벡터 검색 메서드"""
        loop = asyncio.get_event_loop()


        if namespace is None:
            namespace = self._namespace

        print('setting: ', setting)
        if 'no_query_embedding' in setting:
            results = await loop.run_in_executor(
                self.thread_pool,
                functools.partial(
                    self._index.query,
                    vector=[0]*4096,
                    top_k=k,
                    include_metadata=True,
                    namespace=namespace,
                    filter=filter
                )
            )
        else:
            results = await loop.run_in_executor(
                self.thread_pool,
                functools.partial(
                    self._index.query,
                    vector=embedding,
                    top_k=k,
                    include_metadata=True,
                    namespace=namespace,
                    filter=filter
                )
            )

        # text_keys를 미리 가져오기
        text_keys = getattr(self.base_store, "_text_key", [])
        
        # 결과 처리를 실행
        docs = await loop.run_in_executor(
            self.thread_pool,
            lambda: self._process_results(results, text_keys)
        )

        
        
        # docs = []
        # text_keys = getattr(self.base_store, "_text_key", [])
        # if isinstance(text_keys, str):
        #     text_keys = [text_keys]

        # for res in results["matches"]:
        #     metadata = res["metadata"]
        #     id = res.get("id")
        #     if any(k in metadata for k in text_keys):
        #         common_keys = [k for k in text_keys if k in metadata]
        #         text_parts = []

        #         for key in common_keys:
        #             value = metadata.pop(key) if key == "content" else metadata.get(key, "")
        #             # 리스트일 경우 문자열로 변환
        #             if isinstance(value, list):
        #                 value = " ".join(value)
        #             text_parts.append(key + ': ' + value + ' |')

        #         text = " ".join(text_parts)
        #         score = res["score"]
        #         docs.append(
        #             (Document(id=id, page_content='id: ' + id[:-2] + ' |' + text, metadata=metadata), score)
        #         )
        #     else:
        #         print(
        #             f"Found document with no `{self._text_key}` key. Skipping."
        #         )

        print('search_type: ', search_type)
        # 결과 필터링 및 반환도 ThreadPool에서 처리
        if 'summary' in setting:
            return await loop.run_in_executor(
                self.thread_pool,
                self._get_summary_docs,
                docs
            )
        elif setting == 'time_weighted':
            return await loop.run_in_executor(
                self.thread_pool,
                lambda: [(doc, similarity) for doc, similarity in docs if similarity >= score_threshold]
            )
        elif search_type == "similarity_score_threshold":
            return await loop.run_in_executor(
                self.thread_pool,
                lambda: [doc for doc, similarity in docs if similarity >= score_threshold]
            )
        else:
            return docs
        
    def _process_results(self, results: dict, text_keys: List[str]) -> List[Tuple[Document, float]]:
        """결과 처리를 위한 헬퍼 메서드"""
        docs = []
        if isinstance(text_keys, str):
            text_keys = [text_keys]

        for res in results["matches"]:
            metadata = res["metadata"]
            id = res.get("id")
            if any(k in metadata for k in text_keys):
                common_keys = [k for k in text_keys if k in metadata]
                text_parts = []

                for key in common_keys:
                    value = metadata.pop(key) if key == "content" else metadata.get(key, "")
                    if isinstance(value, list):
                        value = " ".join(value)
                    text_parts.append(key + ': ' + value + ' |')

                text = " ".join(text_parts)
                score = res["score"]
                docs.append(
                    (Document(id=id, page_content='id: ' + id[:-2] + ' |' + text, metadata=metadata), score)
                )
            else:
                print(f"Found document with no `{self._text_key}` key. Skipping.")
        
        return docs

    # async def _aget_summary_docs(self, docs: List[Tuple[Document, float]]) -> List[Document]:
    #     """비동기 요약 문서 처리 메서드"""
    #     print('get only summary page_content with async')
    #     summary_docs = docs.copy()
    #     for doc in summary_docs:
    #         doc[0].page_content = doc[0].page_content.split(" <Content>:")[0]
    #     return [doc for doc, _ in summary_docs]

    # # langchain_core의 비동기 인터페이스 구현
    # async def asimilarity_search(
    #     self, query: str, k: int = 4, **kwargs: Any
    # ) -> List[Document]:
    #     """비동기 유사도 검색"""
    #     return await self.base_store.asimilarity_search(query, k=k, **kwargs)

    async def asimilarity_search_with_score(
        self, query: str, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """비동기 점수 포함 유사도 검색"""
        embedding = await self._aembed_query(query)
        combined_kwargs = {}
        if hasattr(self, 'search_kwargs'):
            combined_kwargs.update(self.search_kwargs)
        combined_kwargs.update(kwargs)
        
        return await self.asimilarity_search_by_vector_with_score(
            embedding, **combined_kwargs
        )

    async def _aembed_query(self, query: str) -> List[float]:
        """비동기 쿼리 임베딩"""
        if hasattr(self.base_store._embedding, "aembed_query"):            
            return await self.base_store._embedding.aembed_query(query)
        # 비동기 메서드가 없으면 동기 메서드 사용
        return self.base_store._embedding.embed_query(query)
    
    def __del__(self):
        # 객체가 소멸될 때 thread_pool을 정리
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown()

    # 나머지 메서드는 기본 PineconeVectorStore의 메서드를 호출
    def __getattr__(self, name):
        return getattr(self.base_store, name)

