from langchain_openai import ChatOpenAI
from langchain_upstage import UpstageEmbeddings
from app.core.config import settings


class AIModelManager:
    DEFAULT_MAX_TOKEN = 1024
    DEFAULT_EMBEDDING_MODEL = "embedding-query"
    DEFAULT_LLM_MODEL = "gpt-4o-mini-2024-07-18"
    DEFAULT_LLM_MAX_RETRIES = 2

    def __init__(
        self,
        llm_temperature: float = 0.0,
        llm_max_tokens: int = DEFAULT_MAX_TOKEN,
        llm_time_out: int = None,
        llm_max_retries: int = DEFAULT_LLM_MAX_RETRIES
    ):
        self.embeddings = self._init_embeddings()
        self.llm = self._init_llm(llm_temperature, llm_max_tokens, llm_time_out, llm_max_retries)

    def _init_embeddings(self):
        return UpstageEmbeddings(
            model=self.DEFAULT_EMBEDDING_MODEL,
            api_key=settings.upstage_api_key
        )

    def _init_llm(self, llm_temperature: float, llm_max_tokens: int, llm_time_out: int, llm_max_retries: int):
        return ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model_name=self.DEFAULT_LLM_MODEL,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            timeout=llm_time_out,
            max_retries=llm_max_retries,
            streaming=True
        )

    @staticmethod
    def get_custom_prompt_template():
        return (
            """ 
            The current time is {current_time}.
            You are a highly knowledgeable assistant for question-answering tasks.
            "Based on the following pieces of retrieved context, provide a clear, well-supported,
            and well-structured answer to the question. Summarize key points while including relevant details."
            Make sure your answer utilizes up to the maximum token limit ({MAX_TOKENS} tokens), remaining concise and relevant.
            When referring to a person, use their title based on the most recent data (latest publication_date value).
            Additionally, explain the role or context of the person mentioned in the answer.
            If the answer or the person cannot be determined from the provided context, state clearly that you don't know.
            Respond in Korean.

            우리 교단은 '대한예수교장로회합동'이고 중어서 '합동' 혹은 '합동'이라고 해.
            추상적인 질문을 하면 우리 교단이 기준이야.

            #Question: 
            {input}

            #Context: 
            {context}

            #Answer:
            """
        )

    @staticmethod
    def get_custom_prompt_template_v2():
        return (
            """
            The current time is {current_time}.
            You are a highly knowledgeable assistant calls '카이(KAI)' for question-answering tasks.
            "Based on the following pieces of retrieved context, provide a clear, well-supported,
            and well-structured answer to the question. Summarize key points while including relevant details."
            Make sure your answer utilizes up to the maximum token limit ({MAX_TOKENS} tokens), remaining concise and relevant.
            When referring to a person, use their title based on the most recent data (latest publication_date value).
            Additionally, explain the role or context of the person mentioned in the answer.
            If the answer or the person cannot be checked from the provided context, just say you don't know about question information.
            and answer relation with focus 'Question's word'. like '109회 총회' then focus on '109회' context only.
            Respond in Korean.
            
            우리 교단은 '대한예수교장로회합동'이고 줄여서 '예장합동' 혹은 '합동'이라고 해.
            추상적인 질문을 하면 우리 교단이 기준이야.
            
            #Question: 
            {input}
            
            #Context: 
            {context}
            
            #Answer:
            """
        )
