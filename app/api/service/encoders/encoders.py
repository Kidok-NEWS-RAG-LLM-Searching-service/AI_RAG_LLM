import pickle
from typing import List, Any

from pinecone_text.sparse import BM25Encoder

from app.api.service.tokenizers import KiwiBM25Tokenizer

DEFAULT_ENCODER_MODE = "kiwi"
DEFAULT_ENCODER_LANGUAGE = "english"

import string
from typing import Optional, List
from kiwipiepy import Kiwi

#
# class KiwiBM25Tokenizer:
#     def __init__(self, stop_words: Optional[List[str]] = None):
#         self._tokenizer = self.__initialize_tokenizer()
#         # self.__setup_nltk()
#
#         self._stop_words = set(stop_words) if stop_words else set()
#         self.punctuation = set(string.punctuation)
#
#     @staticmethod
#     def __initialize_tokenizer() -> Kiwi:
#         return Kiwi()
#     #
#     # @staticmethod
#     # def __tokenize(tokenizer: Kiwi, text:str) -> List[str]:
#     #     return [token.form for token in tokenizer.tokenize(text)]
#     #
#     # @staticmethod
#     # def __setup_nltk() -> None:
#     #     try:
#     #         nltk.data.find(DEFAULT_NLTK_TOKENIZER_TYPE)
#     #     except LookupError:
#     #         nltk.download("punkt")
#     #
#     # def __call__(self, text: str) -> List[str]:
#     #     """Make the tokenizer callable."""
#     #     tokens = self.__tokenize(self._tokenizer, text)
#     #     return [token for token in tokens if token not in self._stop_words and token not in self.punctuation]
#     #
#     # def __getstate__(self):
#     #     state = self.__dict__.copy()
#     #     del state["_tokenizer"]
#     #     return state
#     #
#     # def __setstate__(self, state):
#     #     self.__dict__.update(state)
#     #     self._tokenizer = self.__initialize_tokenizer()
#
#
#     def __call__(self, text: str) -> List[str]:
#         tokens = [token.form for token in self._tokenizer.tokenize(text)]
#         return [
#             word.lower()
#             for word in tokens
#             if word not in self.punctuation and word not in self._stop_words
#         ]
#
#     def __getstate__(self):
#         """Pickle로 저장 가능한 상태를 반환합니다."""
#         state = self.__dict__.copy()
#         # _tokenizer는 저장하지 않습니다.
#         del state["_tokenizer"]
#         return state
#
#     def __setstate__(self, state):
#         """Pickle에서 복원된 상태를 설정합니다."""
#         self.__dict__.update(state)
#         # _tokenizer를 새로 초기화합니다.
#         self._tokenizer = self.__initialize_tokenizer()
#

class SparseEncoder:

    @staticmethod
    def create_sparse_encoder(stopwords: List[str], mode: str = DEFAULT_ENCODER_MODE) -> BM25Encoder:
        bm25_encoder = BM25Encoder(language=DEFAULT_ENCODER_LANGUAGE)
        if mode == DEFAULT_ENCODER_MODE:
            bm25_encoder._tokenizer = KiwiBM25Tokenizer(stop_words=stopwords)
        return bm25_encoder

    # def fit(self, bm25_encoder, contents: List[str], save_path: str) -> str:
    #     if bm25_encoder is None:
    #         raise ValueError("Need to initialize bm25 encoder")
    #     bm25_encoder.fit(contents)
    #     with open(save_path, "wb") as f:
    #         pickle.dump(bm25_encoder, f)
    #     return save_path
    #
    # @staticmethod
    # def load(file_path: str) -> Any:
    #     try:
    #         with open(file_path, "rb") as f:
    #             loaded_file = pickle.load(f)
    #         return loaded_file
    #     except Exception as e:
    #         return None

    @staticmethod
    def fit(
                bm25_encoder: BM25Encoder, contents: List[str], save_path: str
    ) -> str:
        """Sparse Encoder를 학습하고 model_path를 제외한 데이터를 저장합니다."""
        import os

        # 디렉토리 생성 확인
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # BM25Encoder 학습
        bm25_encoder.fit(contents)

        # 데이터를 pickle로 저장
        with open(save_path, "wb") as f:
            pickle.dump(bm25_encoder, f)

        print(f"[fit_sparse_encoder] Saved Sparse Encoder to: {save_path}")
        return save_path

    @staticmethod
    def load(file_path: str) -> BM25Encoder:

        """저장된 Sparse Encoder 데이터를 로드하고 복원합니다."""
        try:
            with open(file_path, "rb") as f:
                bm25_encoder = pickle.load(f)
            print(f"[load_sparse_encoder] Loaded Sparse Encoder from: {file_path}")
            return bm25_encoder

        except Exception as e:
            print(f"[load_sparse_encoder] Error: {e}")
            return None

sparse_encoder = SparseEncoder()

