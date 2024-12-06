from typing import List

import requests


DEFAULT_KOREAN_STOP_WORDS_FILE_URL = "https://raw.githubusercontent.com/teddylee777/langchain-teddynote/main/assets/korean_stopwords.txt"


class StopwordsManager:

    def __init__(self, file_url: str = DEFAULT_KOREAN_STOP_WORDS_FILE_URL):
        self.file_url = file_url

    def fetch_stopwords(self) -> List[str]:
        response = requests.get(self.file_url)
        response.raise_for_status()
        stopwords_data = response.text
        return [word.strip() for word in stopwords_data.splitlines()]
