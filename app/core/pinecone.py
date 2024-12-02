from AI_RAG_LLM.app.core.config import settings
from pinecone.grpc import PineconeGRPC as Pinecone

import pickle
import string
from typing import List, Optional
from kiwipiepy import Kiwi
import nltk
import requests


class PineconeConfig:
    pinecone_api_key = settings.get_pinecone_api_key()
    index_name = settings.get_pinecone_index_name()
    namespace = settings.get_pinecone_namespace()

    def __int__(self):
        self.pc = Pinecone(
            api_key=self.pinecone_api_key
        )
        self.index = self.pc.Index(self.index_name)
        self.tokenizer: str = "kiwi"
        self.sparse_encoder_path: str = '' # Sparse Encoder 저장경로(save_path)
        self.top_k: int = 10
        self.alpha: float = 0.5 # alpha=0.75로 설정한 경우, (0.75: Dense Embedding, 0.25: Sparse Embedding)


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

    def bm25(self):
        try:
            with open(self.sparse_encoder_path, "rb") as f:
                bm25 = pickle.load(f)
            if self.tokenizer == "kiwi":
                bm25._tokenizer = KiwiBM25Tokenizer(stop_words=self.stopwords())
            return bm25
        except Exception as e:
            print(e)
            return {}

    def _check_namespace_(self):
        namespace_keys = self.index.describe_index_stats()["namespaces"].keys()
        if self.namespace not in namespace_keys:
            raise ValueError(
                f"'{self.namespace}'를 '{list(namespace_keys)}'에서 찾지 못했습니다."
            )
        return True




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


class KiwiTokenizer:
    def __init__(self):
        self.kiwi = Kiwi()

    def tokenize(self, text, type="list"):
        if type == "list":
            return [token.form for token in self.kiwi.tokenize(text)]
        else:
            return " ".join([token.form for token in self.kiwi.tokenize(text)])



