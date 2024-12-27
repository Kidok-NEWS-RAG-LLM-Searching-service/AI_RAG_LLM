from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from app.core.config import settings
from app.core.llm import AIModelManager
from typing import List, Dict
from langchain_core.embeddings import Embeddings
import pickle
import requests
from app.core.tokenizer import KiwiBM25Tokenizer


class InitVectorStore:
    ai_model_manager = AIModelManager()

    def init_pinecone_vectorstore(self):
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        text_field = "content"

        return PineconeVectorStore(
            index,
            embedding=self.ai_model_manager.embeddings,
            text_key=text_field,
            namespace=''
        )

    def init_customize_vectorstore(self):
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        text_field = ["content", "journalist_name"]

        return PineconeVectorStore(
            index=index,
            embedding=self.ai_model_manager.embeddings,
            text_key=text_field,
            namespace=''
        )
        
    def init_pinecone_index(self,
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

    @staticmethod        
    def stopwords():
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

        
