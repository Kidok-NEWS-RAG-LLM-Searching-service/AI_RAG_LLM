import pickle
from typing import List, Any

from pinecone_text.sparse import BM25Encoder

from app.api.service.tokenizers import KiwiBM25Tokenizer

DEFAULT_ENCODER_MODE = "kiwi"
DEFAULT_ENCODER_LANGUAGE = "english"


class SparseEncoder:

    def create_sparse_encoder(self, stopwords: List[str], mode: str = DEFAULT_ENCODER_MODE) -> BM25Encoder:
        bm25_encoder = BM25Encoder(language=DEFAULT_ENCODER_LANGUAGE)
        if mode == DEFAULT_ENCODER_MODE:
            bm25_encoder._tokenizer = KiwiBM25Tokenizer(stop_words=stopwords)
        return bm25_encoder

    def fit(self, bm25_encoder, contents: List[str], save_path: str) -> str:
        if bm25_encoder is None:
            raise ValueError("Need to initialize bm25 encoder")
        bm25_encoder.fit(contents)
        with open(save_path, "wb") as f:
            pickle.dump(bm25_encoder, f)
        return save_path

    @staticmethod
    def load(file_path: str) -> Any:
        try:
            with open(file_path, "rb") as f:
                loaded_file = pickle.load(f)
            return loaded_file
        except Exception as e:
            return None


sparse_encoder = SparseEncoder()
