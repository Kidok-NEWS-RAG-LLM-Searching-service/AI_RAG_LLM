import string
from typing import Optional, List

from kiwipiepy import Kiwi
import nltk

DEFAULT_NLTK_TOKENIZER_TYPE = "tokenizers/punkt"


class KiwiBM25Tokenizer:
    def __init__(self, stop_words: Optional[List[str]] = None):
        self._tokenizer = self.__initialize_tokenizer()
        # self.__setup_nltk()

        self._stop_words = set(stop_words) if stop_words else set()
        self.punctuation = set(string.punctuation)

    @staticmethod
    def __initialize_tokenizer() -> Kiwi:
        return Kiwi()
    #
    # @staticmethod
    # def __tokenize(tokenizer: Kiwi, text:str) -> List[str]:
    #     return [token.form for token in tokenizer.tokenize(text)]
    #
    # @staticmethod
    # def __setup_nltk() -> None:
    #     try:
    #         nltk.data.find(DEFAULT_NLTK_TOKENIZER_TYPE)
    #     except LookupError:
    #         nltk.download("punkt")
    #
    # def __call__(self, text: str) -> List[str]:
    #     """Make the tokenizer callable."""
    #     tokens = self.__tokenize(self._tokenizer, text)
    #     return [token for token in tokens if token not in self._stop_words and token not in self.punctuation]
    #
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     del state["_tokenizer"]
    #     return state
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     self._tokenizer = self.__initialize_tokenizer()


    def __call__(self, text: str) -> List[str]:
        tokens = [token.form for token in self._tokenizer.tokenize(text)]
        return [
            word.lower()
            for word in tokens
            if word not in self.punctuation and word not in self._stop_words
        ]

    def __getstate__(self):
        """Pickle로 저장 가능한 상태를 반환합니다."""
        state = self.__dict__.copy()
        # _tokenizer는 저장하지 않습니다.
        del state["_tokenizer"]
        return state

    def __setstate__(self, state):
        """Pickle에서 복원된 상태를 설정합니다."""
        self.__dict__.update(state)
        # _tokenizer를 새로 초기화합니다.
        self._tokenizer = self.__initialize_tokenizer()


