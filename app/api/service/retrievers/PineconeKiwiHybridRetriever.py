from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from pydantic import ConfigDict, model_validator, BaseModel
from typing import List, Dict, Any, Optional, Tuple


class PineconeKiwiHybridRetriever(BaseRetriever, BaseModel):

    embeddings: Embeddings
    sparse_encoder: Any
    index: Any
    top_k: int
    alpha: float
    namespace: Optional[str] = None
    pc: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, embeddings: Embeddings, sparse_encoder: Any, index: Any, top_k: int = 10, alpha: float = 1,
                 namespace: Optional[str] = None, pc: Any = None, *args: Any, **kwargs: Any):
        super().__init__(
            embeddings=embeddings,
            sparse_encoder=sparse_encoder,
            index=index,
            top_k=top_k,
            alpha=alpha
        )
        self.embeddings = embeddings
        self.sparse_encoder = sparse_encoder
        self.index = index
        self.top_k = top_k
        self.alpha = alpha
        self.namespace = namespace
        self.pc = pc

    @model_validator(mode="after")
    def validate_environment(self, values: Dict) -> Dict:
        try:
            from pinecone_text.hybrid import hybrid_convex_scale
            from pinecone_text.sparse.base_sparse_encoder import BaseSparseEncoder
        except ImportError:
            raise ImportError(
                "Could not import pinecone_text python package. "
                "Please install it with `pip install pinecone_text`."
            )
        return values

    def update_params(self, top_k: Optional[int] = None, alpha: Optional[float] = None):
        if top_k is not None:
            self.top_k = top_k
        if alpha is not None:
            self.alpha = alpha

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **search_kwargs,
    ) -> List[Document]:
        alpha = self._get_alpha(search_kwargs)
        dense_vec, sparse_vec = self._encode_query(query, alpha)
        query_params = self._build_query_params(
            dense_vec, sparse_vec, search_kwargs, include_metadata=True
        )
        query_response = self.index.query(**query_params)
        documents = self._process_query_response(query_response)

        if (
            "search_kwargs" in search_kwargs
            and "rerank" in search_kwargs["search_kwargs"]
        ):
            documents = self._rerank_documents(query, documents, **search_kwargs)

        return documents

    def _get_alpha(self, search_kwargs: Dict[str, Any]) -> float:
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

        return query_params

    @staticmethod
    def _process_query_response(query_response: Dict[str, Any]) -> List[Document]:
        return [
            Document(page_content=r.metadata["content"], metadata=r.metadata)
            for r in query_response["matches"]
        ]

    def _rerank_documents(
        self, query: str, documents: List[Document], **kwargs
    ) -> List[Document]:
        options = kwargs.get("search_kwargs", {})
        rerank_model = options.get("rerank_model", "bge-reranker-v2-m3")
        top_n = options.get("top_n", len(documents))
        rerank_docs = [
            {"id": str(i), "text": doc.page_content} for i, doc in enumerate(documents)
        ]

        if self.pc is not None:
            re_ranked_result = self.pc.inference.rerank(
                model=rerank_model,
                query=query,
                documents=rerank_docs,
                top_n=top_n,
                return_documents=True,
            )

            re_ranked_documents = []

            for item in re_ranked_result.data:
                original_doc = documents[int(item["index"])]
                re_ranked_doc = Document(
                    page_content=original_doc.page_content,
                    metadata={**original_doc.metadata, "rerank_score": item["score"]},
                )
                re_ranked_documents.append(re_ranked_doc)

            return re_ranked_documents
        else:
            raise ValueError("Pinecone 인덱스가 초기화되지 않았습니다.")
