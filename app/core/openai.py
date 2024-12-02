from AI_RAG_LLM.app.core.config import settings
from langchain_openai import ChatOpenAI

class OpenaiConfig:
    MODEL_NAME = 'gpt-4o-mini-2024-07-18'
    MAX_TOKENS = 1024
    TEMPERATURE = 0.0
    TIMEOUT = 10
    MAX_RETRIES = 3
    openai_api_key = settings.get_openai_api_key()

    def __int__(self):
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model_name=self.MODEL_NAME,
            temperature=self.TEMPERATURE,
            max_tokens=self.MAX_TOKENS,
            timeout=self.TIMEOUT,
            max_retries=self.MAX_RETRIES
        )













