import dill as pickle

from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_core.embeddings import Embeddings
from typing import List, Dict, Any, Optional

from pinecone_text.sparse import BM25Encoder

from app.api.service.encoders.encoders import sparse_encoder
from app.api.service.tokenizers import KiwiBM25Tokenizer
from app.core.config import settings


class PineconeIndexInitializer:
    def __init__(
        self,
        sparse_encoder_path: str,
        stopwords: Optional[List[str]] = None,
        tokenizer: str = "kiwi",
        embeddings: Optional[Embeddings] = None,
        top_k: int = 10,
        alpha: float = 1,
    ):
        self.api_key = settings.pinecone_api_key
        self.namespace = settings.pinecone_namespace
        self.index_name = settings.pinecone_index_name

        self.sparse_encoder_path = sparse_encoder_path
        self.stopwords = stopwords
        self.tokenizer = tokenizer
        self.embeddings = embeddings
        self.top_k = top_k
        self.alpha = alpha

        self.pinecone = Pinecone(api_key=self.api_key)
        self.index = None

        # for pinecone params
        self.init_data = self.__initialize_index(index_name=self.index_name, namespace=self.namespace)

    def get_pinecone_init_data(self):
        return self.init_data

    def __initialize_index(self, index_name: str, namespace: str) -> Dict[str, Any]:
        self.index = self.pinecone.Index(index_name)
        print(f"[initialize_index]\n{self.index.describe_index_stats()}")

        self._load_sparse_encoder()
        self._set_tokenizer()

        namespace_keys = self.index.describe_index_stats()["namespaces"].keys()
        if namespace not in namespace_keys:
            raise ValueError(
                f"`{namespace}` not found in available namespaces: {list(namespace_keys)}"
            )

        return {
            "index": self.index,
            "namespace": namespace,
            "sparse_encoder": self.sparse_encoder,
            "embeddings": self.embeddings,
            "top_k": self.top_k,
            "alpha": self.alpha,
            "pinecone": self.pinecone,
        }

    def _load_sparse_encoder(self):
        print(self.sparse_encoder_path)
        """Load the sparse encoder from the specified path."""

        try:
            with open(self.sparse_encoder_path, "rb") as f:
                print("Start loading sparse encoder")
                self.sparse_encoder = pickle.load(f)
                print("Sparse encoder loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load sparse encoder: {e}")

    def _set_tokenizer(self):
        if self.tokenizer == "kiwi":
            try:
                print("Start setting tokenizer")
                self.sparse_encoder._tokenizer = KiwiBM25Tokenizer(stop_words=self.stopwords)
                print("Tokenizer set successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to set tokenizer: {e}")
        else:
            print(f"Tokenizer '{self.tokenizer}' is not supported.")

    def __describe_index(self) -> Dict[str, Any]:
        if self.index is None:
            raise RuntimeError("Index has not been initialized.")
        return self.index.describe_index_stats()

