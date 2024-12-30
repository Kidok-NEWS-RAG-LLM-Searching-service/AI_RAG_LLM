from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from pydantic import ConfigDict, model_validator
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from functools import partial

class NewPineconeKiwiHybridRetriever(BaseRetriever):
    """
    Pinecone과 Kiwi를 결합한 하이브리드 검색기 클래스입니다.

    이 클래스는 밀집 벡터와 희소 벡터를 모두 사용하여 문서를 검색합니다.
    Pinecone 인덱스와 Kiwi 토크나이저를 활용하여 효과적인 하이브리드 검색을 수행합니다.

    매개변수:
        embeddings (Embeddings): 문서와 쿼리를 밀집 벡터로 변환하는 임베딩 모델
        sparse_encoder (Any): 문서와 쿼리를 희소 벡터로 변환하는 인코더 (예: BM25Encoder)
        index (Any): 검색에 사용할 Pinecone 인덱스 객체
        top_k (int): 검색 결과로 반환할 최대 문서 수 (기본값: 10)
        alpha (float): 밀집 벡터와 희소 벡터의 가중치를 조절하는 파라미터 (0 에서 1 사이, 기본값: 0.5),  alpha=0.75로 설정한 경우, (0.75: Dense Embedding, 0.25: Sparse Embedding)
        namespace (Optional[str]): Pinecone 인덱스 내에서 사용할 네임스페이스 (기본값: None)
    """

    embeddings: Embeddings
    sparse_encoder: Any
    index: Any
    top_k: int = 10
    alpha: float = 0.5
    namespace: Optional[str] = None
    pc: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_environment(cls, values: Dict) -> Dict:
        """
        필요한 패키지가 설치되어 있는지 확인하는 메서드입니다.

        Returns:
            Dict: 유효성 검사를 통과한 값들의 딕셔너리
        """
        try:
            from pinecone_text.hybrid import hybrid_convex_scale
            from pinecone_text.sparse.base_sparse_encoder import BaseSparseEncoder
        except ImportError:
            raise ImportError(
                "Could not import pinecone_text python package. "
                "Please install it with `pip install pinecone_text`."
            )
        return values

    def similarity_search(
        self, 
        query: str, 
        k: int = None, 
        filter: Optional[dict] = None,
        **kwargs
    ) -> List[Document]:
        """
        주어진 쿼리에 대해 유사한 문서를 검색하는 메서드입니다.

        Args:
            query (str): 검색 쿼리
            k (int, optional): 반환할 문서 수
            filter (dict, optional): 검색 필터
            **kwargs: 추가 검색 매개변수

        Returns:
            List[Document]: 검색된 문서 리스트
        """
        search_kwargs = {"search_kwargs": {}}
        
        if k is not None:
            search_kwargs["search_kwargs"]["top_k"] = k
        if filter is not None:
            search_kwargs["search_kwargs"]["filter"] = filter
        
        return self._get_relevant_documents(
            query,
            run_manager=CallbackManagerForRetrieverRun.get_noop_manager(),
            **search_kwargs
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **search_kwargs,
    ) -> List[Document]:
        """
        주어진 쿼리에 대해 관련 문서를 검색하는 메인 메서드입니다.

        Args:
            query (str): 검색 쿼리
            run_manager (CallbackManagerForRetrieverRun): 콜백 관리자
            **search_kwargs: 추가 검색 매개변수

        Returns:
            List[Document]: 관련 문서 리스트
        """
        alpha = self._get_alpha(search_kwargs)
        dense_vec, sparse_vec = self._encode_query(query, alpha)
        query_params = self._build_query_params(
            dense_vec, sparse_vec, search_kwargs, include_metadata=True
        )

        query_response = self.index.query(**query_params)
        # print("namespace", self.namespace)

        documents = self._process_query_response(query_response)

        # Rerank 옵션이 있는 경우 rerank 수행
        if (
            "search_kwargs" in search_kwargs
            and "rerank" in search_kwargs["search_kwargs"]
        ):
            documents = self._rerank_documents(query, documents, **search_kwargs)

        return documents

    def _get_alpha(self, search_kwargs: Dict[str, Any]) -> float:
        """
        알파 값을 가져오는 메서드입니다.

        Args:
            search_kwargs (Dict[str, Any]): 검색 매개변수

        Returns:
            float: 알파 값
        """
        if (
            "search_kwargs" in search_kwargs
            and "alpha" in search_kwargs["search_kwargs"]
        ):
            return search_kwargs["search_kwargs"]["alpha"]
        return self.alpha

    def _encode_query(
        self, query: str, alpha: float
    ) -> Tuple[List[float], Dict[str, Any]]:
        from pinecone_text.hybrid import hybrid_convex_scale
        """
        쿼리를 인코딩하는 메서드입니다.

        Args:
            query (str): 인코딩할 쿼리
            alpha (float): 하이브리드 스케일링에 사용할 알파 값

        Returns:
            Tuple[List[float], Dict[str, Any]]: 밀집 벡터와 희소 벡터의 튜플
        """
        sparse_vec = self.sparse_encoder.encode_queries(query)
        dense_vec = self.embeddings.embed_query(query)
        dense_vec, sparse_vec = hybrid_convex_scale(dense_vec, sparse_vec, alpha=alpha)
        sparse_vec["values"] = [float(s1) for s1 in sparse_vec["values"]]
        return dense_vec, sparse_vec

    def _build_query_params(
        self,
        dense_vec: List[float],
        sparse_vec: Dict[str, Any],
        search_kwargs: Dict[str, Any],
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        쿼리 파라미터를 구성하는 메서드입니다.

        Args:
            dense_vec (List[float]): 밀집 벡터
            sparse_vec (Dict[str, Any]): 희소 벡터
            search_kwargs (Dict[str, Any]): 검색 매개변수
            include_metadata (bool): 메타데이터 포함 여부

        Returns:
            Dict[str, Any]: 구성된 쿼리 파라미터
        """
        query_params = {
            "vector": dense_vec,
            "sparse_vector": sparse_vec,
            "top_k": self.top_k,
            "include_metadata": include_metadata,
            "namespace": self.namespace,
        }

        if "search_kwargs" in search_kwargs:
            kwargs = search_kwargs["search_kwargs"]
            query_params.update(
                {
                    "filter": kwargs.get("filter", query_params.get("filter")),
                    "top_k": kwargs.get("top_k")
                    or kwargs.get("k", query_params["top_k"]),
                }
            )
        
            print('filter: ', kwargs.get("filter", query_params.get("filter")))

        return query_params

    def _process_query_response(self, query_response: Dict[str, Any]) -> List[Document]:
        """
        쿼리 응답을 처리하는 메서드입니다.

        Args:
            query_response (Dict[str, Any]): Pinecone 쿼리 응답

        Returns:
            List[Document]: 처리된 문서 리스트
        """
        return [
            Document(id=r.metadata["id"], page_content='id: '+r.metadata["id"][:-2]+' | '+ r.metadata["content"], metadata=r.metadata)
            for r in query_response["matches"]
        ]

    def _rerank_documents(
        self, query: str, documents: List[Document], **kwargs
    ) -> List[Document]:
        """
        검색된 문서를 재정렬하는 메서드입니다.

        Args:
            query (str): 검색 쿼리
            documents (List[Document]): 재정렬할 문서 리스트
            **kwargs: 추가 매개변수

        Returns:
            List[Document]: 재정렬된 문서 리스트
        """
        # print("[rerank_documents]")
        options = kwargs.get("search_kwargs", {})
        rerank_model = options.get("rerank_model", "bge-reranker-v2-m3")
        top_n = options.get("top_n", len(documents))
        rerank_docs = [
            {"id": str(i), "text": doc.page_content} for i, doc in enumerate(documents)
        ]

        if self.pc is not None:
            reranked_result = self.pc.inference.rerank(
                model=rerank_model,
                query=query,
                documents=rerank_docs,
                top_n=top_n,
                return_documents=True,
            )

            # 재정렬된 결과를 기반으로 문서 리스트 재구성
            reranked_documents = []

            for item in reranked_result.data:
                original_doc = documents[int(item["index"])]
                reranked_doc = Document(
                    page_content=original_doc.page_content,
                    metadata={**original_doc.metadata, "rerank_score": item["score"]},
                )
                reranked_documents.append(reranked_doc)

            return reranked_documents
        else:
            raise ValueError("Pinecone 인덱스가 초기화되지 않았습니다.")

    # async def ainvoke(self, query: str, **kwargs) -> List[Document]:
    #     search_kwargs = kwargs.get('search_kwargs', {})
    #     filter_dict = search_kwargs.get('filter', {})
        
    #     return await self.asimilarity_search(
    #         query,
    #         filter=filter_dict  # filter를 직접 전달
    #     )
    
    async def asimilarity_search(
        self, 
        query: str, 
        k: int = None, 
        filter: Optional[dict] = None,
        **kwargs
    ) -> List[Document]:
        """
        주어진 쿼리에 대해 유사한 문서를 비동기적으로 검색하는 메서드입니다.

        Args:
            query (str): 검색 쿼리
            k (int, optional): 반환할 문서 수
            filter (dict, optional): 검색 필터
            **kwargs: 추가 검색 매개변수

        Returns:
            List[Document]: 검색된 문서 리스트
        """
        search_kwargs = {"search_kwargs": {}}
        
        if k is not None:
            search_kwargs["search_kwargs"]["top_k"] = k
        if filter is not None:
            search_kwargs["search_kwargs"]["filter"] = filter
        
        return await self._aget_relevant_documents(
            query,
            run_manager=CallbackManagerForRetrieverRun.get_noop_manager(),
            **search_kwargs
        )
        
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **search_kwargs,
    ) -> List[Document]:
        """
        주어진 쿼리에 대해 관련 문서를 비동기적으로 검색하는 메인 메서드입니다.

        Args:
            query (str): 검색 쿼리
            run_manager (CallbackManagerForRetrieverRun): 콜백 관리자
            **search_kwargs: 추가 검색 매개변수

        Returns:
            List[Document]: 관련 문서 리스트
        """


        # 알파 값 가져오기
        alpha = self._get_alpha(search_kwargs)
        
        # 쿼리 인코딩 (CPU 작업이므로 ThreadPoolExecutor에서 실행)
        loop = asyncio.get_event_loop()
        encode_func = partial(self._encode_query, query, alpha)
        dense_vec, sparse_vec = await loop.run_in_executor(None, encode_func)
        
        # 쿼리 파라미터 구성
        query_params = self._build_query_params(
            dense_vec, sparse_vec, search_kwargs, include_metadata=True
        )

        # Pinecone 쿼리를 별도 스레드에서 실행 (병렬 처리)
        query_response = await loop.run_in_executor(
            None, 
            partial(self.index.query, **query_params)
        )

        # 쿼리 응답 처리
        documents = self._process_query_response(query_response)

        # Rerank 옵션이 있는 경우 rerank 수행
        if (
            "search_kwargs" in search_kwargs
            and "rerank" in search_kwargs["search_kwargs"]
        ):
            documents = await self._arerank_documents(query, documents, **search_kwargs)

        return documents

    async def _arerank_documents(
        self, 
        query: str, 
        documents: List[Document], 
        **kwargs
    ) -> List[Document]:
        """
        검색된 문서를 비동기적으로 재정렬하는 메서드입니다.

        Args:
            query (str): 검색 쿼리
            documents (List[Document]): 재정렬할 문서 리스트
            **kwargs: 추가 매개변수

        Returns:
            List[Document]: 재정렬된 문서 리스트
        """
        options = kwargs.get("search_kwargs", {})
        rerank_model = options.get("rerank_model", "bge-reranker-v2-m3")
        top_n = options.get("top_n", len(documents))
        rerank_docs = [
            {"id": str(i), "text": doc.page_content} for i, doc in enumerate(documents)
        ]

        if self.pc is not None:
            # Pinecone rerank API 호출 (I/O 작업이므로 비동기로 처리)
            loop = asyncio.get_event_loop()
            reranked_result = await loop.run_in_executor(
                None,
                partial(
                    self.pc.inference.rerank,
                    model=rerank_model,
                    query=query,
                    documents=rerank_docs,
                    top_n=top_n,
                    return_documents=True,
                )
            )

            # 재정렬된 결과를 기반으로 문서 리스트 재구성
            reranked_documents = []
            for item in reranked_result.data:
                original_doc = documents[int(item["index"])]
                reranked_doc = Document(
                    page_content=original_doc.page_content,
                    metadata={**original_doc.metadata, "rerank_score": item["score"]},
                )
                reranked_documents.append(reranked_doc)

            return reranked_documents
        else:
            raise ValueError("Pinecone 인덱스가 초기화되지 않았습니다.")

    async def asimilarity_search(
        self, 
        query: str, 
        k: int = None, 
        filter: Optional[dict] = None,
        **kwargs
    ) -> List[Document]:
        """
        주어진 쿼리에 대해 유사한 문서를 비동기적으로 검색하는 메서드입니다.

        Args:
            query (str): 검색 쿼리
            k (int, optional): 반환할 문서 수
            filter (dict, optional): 검색 필터
            **kwargs: 추가 검색 매개변수

        Returns:
            List[Document]: 검색된 문서 리스트
        """
        search_kwargs = {"search_kwargs": {}}
        
        if k is not None:
            search_kwargs["search_kwargs"]["top_k"] = k
        if filter is not None:
            search_kwargs["search_kwargs"]["filter"] = filter
        
        return await self._aget_relevant_documents(
            query,
            run_manager=CallbackManagerForRetrieverRun.get_noop_manager(),
            **search_kwargs
        )

