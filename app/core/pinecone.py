import settings
import string
from typing import List, Optional
from kiwipiepy import Kiwi
import nltk
import pickle
from pinecone_text.sparse import BM25Encoder
from typing import List, Any
import pickle
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_core.embeddings import Embeddings
from typing import List, Dict, Any, Optional, Tuple
import requests
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from pydantic import ConfigDict, model_validator
from typing import List, Dict, Any, Optional, Tuple


class PineconeConfig:
    SPARSE_ENCODER_PTAH = "./yong_contextual_sparse_encoder.pkl"
    pinecone_api_key = settings.get_pinecone_api_key()
    index_name = settings.get_pinecone_index_name()
    namespace = settings.get_pinecone_namespace()

    def __int__(self):
        self.pc = Pinecone(
            api_key=self.pinecone_api_key
        )


    def stopwords(self):
        # GitHub URL로부터 'korean_stopwords.txt' 파일을 읽어 한국어 불용어
        file_url = "https://raw.githubusercontent.com/teddylee777/langchain-teddynote/main/assets/korean_stopwords.txt"

        # 불용어 파일을 인터넷에서 가져옵니다.
        response = requests.get(file_url)
        response.raise_for_status()  # HTTP 요청이 실패하면 예외를 발생시킵니다.

        # 응답으로부터 텍스트 데이터를 받아옵니다.
        stopwords_data = response.text

        # 텍스트 데이터를 줄 단위로 분리합니다.
        stopwords = stopwords_data.splitlines()

        # 각 줄에서 여분의 공백 문자(개행 문자 등)를 제거합니다.
        return [word.strip() for word in stopwords]


    def create_sparse_encoder(self, stopwords: List[str], mode: str = "kiwi") -> BM25Encoder:
        """BM25Encoder를 생성하고 반환합니다."""
        bm25 = BM25Encoder(language="english")
        if mode == "kiwi":
            bm25._tokenizer = KiwiBM25Tokenizer(stop_words=stopwords)
        return bm25

    def fit_sparse_encoder(
            self, sparse_encoder: BM25Encoder, contents: List[str], save_path: str
    ) -> str:
        """Sparse Encoder 를 학습하고 저장합니다."""
        sparse_encoder.fit(contents)
        with open(save_path, "wb") as f:
            pickle.dump(sparse_encoder, f)
        print(f"[fit_sparse_encoder]\nSaved Sparse Encoder to: {save_path}")
        return save_path

    def load_sparse_encoder(self, file_path: str) -> Any:
        """저장된 스파스 인코더를 로드합니다."""
        try:
            with open(file_path, "rb") as f:
                loaded_file = pickle.load(f)
            print(f"[load_sparse_encoder]\nLoaded Sparse Encoder from: {file_path}")
            return loaded_file
        except Exception as e:
            print(f"[load_sparse_encoder]\n{e}")
            return None

    def init_pinecone_index(
            self,
            index_name: str,
            namespace: str,
            api_key: str,
            sparse_encoder_path: str = None,
            stopwords: List[str] = None,
            tokenizer: str = "kiwi",
            embeddings: Embeddings = None,
            top_k: int = 10,
            alpha: float = 0.5,
    ) -> Dict:
        """Pinecone 인덱스를 초기화하고 필요한 구성 요소를 반환합니다."""
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        print(f"[init_pinecone_index]\n{index.describe_index_stats()}")

        try:
            with open(sparse_encoder_path, "rb") as f:
                print('start load')
                bm25 = pickle.load(f)
                print('finish load')
            if tokenizer == "kiwi":
                print('start tokenizer')
                bm25._tokenizer = KiwiBM25Tokenizer(stop_words=stopwords)
                print('finish tokenizer')
        except Exception as e:
            print(e)
            return {}

        namespace_keys = index.describe_index_stats()["namespaces"].keys()
        if namespace not in namespace_keys:
            raise ValueError(
                f"`{namespace}` 를 `{list(namespace_keys)}` 에서 찾지 못했습니다."
            )

        return {
            "index": index,
            "namespace": namespace,
            "sparse_encoder": bm25,
            "embeddings": embeddings,
            "top_k": top_k,
            "alpha": alpha,
            "pc": pc,
        }


class KiwiBM25Tokenizer:
    def __init__(self, stop_words: Optional[List[str]] = None):
        self._setup_nltk()
        self._stop_words = set(stop_words) if stop_words else set()
        self._punctuation = set(string.punctuation)
        self._tokenizer = self._initialize_tokenizer()

    @staticmethod
    def _initialize_tokenizer() -> Kiwi:
        return Kiwi()

    @staticmethod
    def _tokenize(tokenizer: Kiwi, text: str) -> List[str]:
        return [token.form for token in tokenizer.tokenize(text)]

    @staticmethod
    def _setup_nltk() -> None:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    def __call__(self, text: str) -> List[str]:
        tokens = self._tokenize(self._tokenizer, text)
        return [
            word.lower()
            for word in tokens
            if word not in self._punctuation and word not in self._stop_words
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_tokenizer"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokenizer = self._initialize_tokenizer()


class PineconeKiwiHybridRetriever(BaseRetriever):
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
            Document(page_content=r.metadata["context"], metadata=r.metadata)
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

