import string
from typing import Optional, List

from kiwipiepy import Kiwi
import nltk

DEFAULT_NLTK_TOKENIZER_TYPE = "tokenizers/punkt"


class KiwiBM25Tokenizer:
    def __init__(self, stop_words: Optional[List[str]] = None):
        self._tokenizer = self.__initialize_tokenizer()
        self.__setup_nltk()

        self._stop_words = set(stop_words) if stop_words else set()
        self.punctuation = set(string.punctuation)

    @staticmethod
    def __initialize_tokenizer() -> Kiwi:
        return Kiwi()

    @staticmethod
    def __tokenize(tokenizer: Kiwi, text:str) -> List[str]:
        return [token.form for token in tokenizer.tokenize(text)]

    @staticmethod
    def __setup_nltk() -> None:
        try:
            nltk.data.find(DEFAULT_NLTK_TOKENIZER_TYPE)
        except LookupError:
            nltk.download("punkt")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_tokenizer"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokenizer = self.__initialize_tokenizer()
