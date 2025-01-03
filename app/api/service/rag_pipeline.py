import os
from datetime import datetime, timedelta
from typing import List
import json
import re
from functools import wraps
import time
import numpy as np

import pandas as pd
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app.api.repository.ai_model_performance_repository import ai_model_performance_log_repository
from app.api.repository.ai_model_performance_type import AIModelPerformanceType
from app.api.service.encoders.encoders import sparse_encoder
from app.api.service.logs.log import put_search_response_tracking
from app.api.service.managers.stop_words_manager import StopwordsManager
from app.api.service.retrievers.NewPineconeKiwiHybridRetriever import NewPineconeKiwiHybridRetriever
from app.api.service.retrievers.PineconeKiwiHybridRetriever import PineconeKiwiHybridRetriever
from app.api.service.retrievers.TimeWeightedCustomVectorStoreRetriever import TimeWeightedCustomVectorStoreRetriever
from app.api.service.retrievers.TimeWeightedJounaralistFilteringVectorStoreRetriever import TimeWeightedJounaralistFilteringVectorStoreRetriever
from app.core.llm import AIModelManager
from app.core.pinecone_index_initializer import PineconeIndexInitializer
from app.core import prompts
from app.core.init_method import InitVectorStore
from app.core.vectorstore import CustomPineconeVectorStore
from app.core.config import settings

from typing import AsyncGenerator


current_dir = os.path.dirname(os.path.abspath(__file__))


class RagPipeline:
    ai_model_manager = AIModelManager()
    
    summary_prompt = ChatPromptTemplate.from_messages(prompts.summary_prompt_template_id_2())
    journalist_prompt = ChatPromptTemplate.from_messages(prompts.journalist_prompt_template_id_5())
    init_vectorstore = InitVectorStore()

    def __init__(self):
      
        self.llm = self.ai_model_manager.llm
        self.embeddings = self.ai_model_manager.embeddings
        self.client = self.ai_model_manager.client   
        self.async_client = self.ai_model_manager.async_client
        
        self.vectorstore = self.init_vectorstore.init_pinecone_vectorstore()
        self.add_jounaralist_name_customize_vectorstore = self.init_vectorstore.init_customize_vectorstore()
        
        self.custom_vectorstore = CustomPineconeVectorStore(base_store=self.vectorstore)
        self.jounaralist_customize_vectorstore = CustomPineconeVectorStore(base_store=self.add_jounaralist_name_customize_vectorstore)
        
        self.timeweighted_retriever = self._init_timeweighted_retriever()
        self.hybird_retriever = self.hybird_dense_sparse_retriever()
        
        # 기본 체인들 미리 생성
        self.summary_chain = create_stuff_documents_chain(self.llm, self.summary_prompt)
        self.journalist_chain = create_stuff_documents_chain(self.llm, self.journalist_prompt)
        
        self.question_answer_chain = self._init_question_answer_chain()
        self.timeweighted_rag_chain = create_retrieval_chain(
            self.timeweighted_retriever, 
            self.question_answer_chain
        )
        
        self.hybrid_rag_chain = create_retrieval_chain(
            self.hybird_retriever, 
            self.question_answer_chain
        )
        
        

    # stop_words_manager = StopwordsManager()



    # sparse_encoder_path = os.path.join("./app/news_rag_llm/yong_contextual_sparse_encoder.pkl")
    # sparse_encoder_path = os.path.join("./app/sparse_encoder_folder/sparse_encoder_1_57000.pkl")
    # sparse_encoder_path = os.path.join("./app/sparse_encoder_folder/sparse_encoder_10000_20000.pkl")
    sparse_encoder_path = os.path.join("./app/sparse_encoder_folder/sparse_encoder_241226.pkl")
    # global_source_set = set()

    # if not os.path.exists(sparse_encoder_path):
    #     print(f"{sparse_encoder_path} not found. Creating sparse encoder...")
    #     contextual_chunks_df = pd.read_parquet("./app/assets/contents_1_80000.parquet", engine="pyarrow")
    #     sparse_encoder_value = sparse_encoder.create_sparse_encoder(
    #         stop_words_manager.fetch_stopwords(), mode="kiwi"
    #     )
    #     saved_path = sparse_encoder.fit(
    #         bm25_encoder=sparse_encoder_value,
    #         contents=contextual_chunks_df.contexts.tolist(),
    #         save_path=sparse_encoder_path
    #     )
    #     print(f"Sparse encoder saved at: {saved_path}")

    # pinecone_index_initializer = PineconeIndexInitializer(
    #     # sparse_encoder_path="./app/news_rag_llm/yong_contextual_sparse_encoder.pkl",
    #     # sparse_encoder_path="./app/sparse_encoder_folder/sparse_encoder_1_57000.pkl",
    #     sparse_encoder_path="./app/sparse_encoder_folder/sparse_encoder_10000_20000.pkl",
    #     stopwords=stop_words_manager.fetch_stopwords(),  # 불용어 사전
    #     tokenizer="kiwi",
    #     embeddings=embeddings,
    #     top_k=20,
    #     alpha=0.5,
    # )

    # init_data = pinecone_index_initializer.get_pinecone_init_data()

    # pinecone_retriever = PineconeKiwiHybridRetriever(
    #     embeddings=init_data["embeddings"],
    #     sparse_encoder=init_data["sparse_encoder"],
    #     index=init_data["index"],
    #     top_k=init_data["top_k"],
    #     alpha=init_data["alpha"],
    #     namespace=init_data["namespace"]
    # )



    def _init_pinecone_index(self):
        index_params = self.init_vectorstore.init_pinecone_index(
        index_name=settings.pinecone_index_name,
        namespace="",
        api_key=settings.pinecone_api_key,
        sparse_encoder_path=self.sparse_encoder_path,
        stopwords=self.init_vectorstore.stopwords(),
        tokenizer="kiwi",
        embeddings=self.embeddings,
        top_k=20,
        alpha=.5,
        )
        
        # 필수 파라미터들이 있는지 확인
        required_params = {
            "embeddings": self.embeddings,
            "sparse_encoder": index_params.get("sparse_encoder"),
            "index": index_params.get("index"),
            "top_k": index_params.get("top_k"),
            "alpha": index_params.get("alpha"),
            "namespace": index_params.get("namespace")
        }
        # return InitVectorStore.init_pinecone_index(
        #     index_name=settings.pinecone_index_name,  # Pinecone 인덱스 이름
        #     namespace="",  # Pinecone Namespace
        #     api_key= settings.pinecone_api_key,  # Pinecone API Key
        #     sparse_encoder_path=self.sparse_encoder_path,  # Sparse Encoder 저장경로(save_path)
        #     stopwords=InitVectorStore.stopwords(),  # 불용어 사전
        #     tokenizer="kiwi",
        #     embeddings=self.embeddings,  # Dense Embedder
        #     top_k=20,  # Top-K 문서 반환 개수
        #     alpha=.3,  # alpha=0.75로 설정한 경우, (0.75: Dense Embedding, 0.25: Sparse Embedding)
        # )    
        return required_params
        
    def hybird_dense_sparse_retriever(self):
        pinecone_params = self._init_pinecone_index()
        return NewPineconeKiwiHybridRetriever(**pinecone_params)


    def _init_timeweighted_retriever(self):
        # Retriever 초기화
        return TimeWeightedCustomVectorStoreRetriever(
            vectorstore=self.custom_vectorstore,
            decay_rate=0.000_001,  # 0.000_000_1
            # k=20,  # 반환할 최대 문서 개수
            search_kwargs={
                'search_type': 'similarity_score_threshold',
                'score_threshold': 0.319,  # 0과 1 사이의 값 설정
                'filter': {'section': {'$nin': ['기독AD']}},
                'setting': 'time_weighted'
            }
        )

    def _init_date_filter_score_retriever(self, date_list: list):
        return self.custom_vectorstore.as_retriever(
            search_kwargs={
                'search_type': 'similarity_score_threshold',
                'score_threshold': 0.319,  # 원래 0.67이였음(similarity_search_with_relevance_scores 계산 방식으로 정규화가 되기 때문에)
                # 그런데 내가 해당 함수 cumstomize하면서 그냥 정규화 안된 score로 점수 거르게 만들어서 이렇게 점수 사용.
                'k': 20,
                "filter": {
                    "init_date": {"$in": date_list},
                    "section": {"$nin": ['설교', '기독AD', '오피니언']}
                },
            }
        )

    def _init_summary_filter_retriever(self, date_list: list):
        return self.custom_vectorstore.as_retriever(
            search_kwargs={
                'k': 50,
                "filter": {
                    "init_date": {"$in": date_list},
                    "section": {"$nin": ['설교', '기독AD', '오피니언']}
                },
                'setting': "summary and no_query_embedding"
            }
        )

    def _init_jounaralist_time_filter_retriever(self, name_list: list, type: str):
        if type == 'recent':
            date = datetime.now().year-1
            setting = 'no_query_embedding'
            decay_rate=-0.01
        else:
            date = 1900
            setting = 'original'
            decay_rate=0.000_000_1
        return TimeWeightedJounaralistFilteringVectorStoreRetriever(
            vectorstore=self.jounaralist_customize_vectorstore,
            decay_rate=decay_rate,
            k=20,  # 반환할 최대 문서 개수
            search_kwargs={
                'name_list': name_list,
                'filter': {
                    'section': {'$nin': ['기독AD']},
                    'init_year': {'$gte': date}
                },
                'type': type,
                'setting': setting
            }
        )


    # prompt = ChatPromptTemplate.from_template(AIModelManager.get_custom_prompt_template_v2())
    # question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # rag_chain = create_retrieval_chain(pinecone_retriever, question_answer_chain)

    # query routing을 위한 LLM 호출하는 함수
    async def query_llm(self, query: str) -> str:
        """
        Query the LLM with a prompt and return the response.
        """
        response = await self.async_client.chat.completions.create(
            model=self.ai_model_manager.DEFAULT_LLM_MODEL,
            messages=[{"role": "system", "content": prompts.query_routing_prompt_system()},
                      {"role": "user", "content": prompts.query_routing_prompt_user(query=query)}]
        )

        return response.choices[0].message.content

    # 날짜 계산에 대한 LLM을 호출하는 함수
    async def date_cal_llm(self, query: str) -> str:
        """
        Query the LLM with a prompt and return the response.
        """
        response = await self.async_client.chat.completions.create(
            model=self.ai_model_manager.DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": prompts.date_cal_prompt_system()},
                {"role": "user", "content": prompts.date_cal_prompt_user_3(query=query)}]
        )

        return response.choices[0].message.content

    # 날짜 계산에 대한 LLM을 호출하는 함수
    async def session_cal_llm(self, query:str) -> str:
        """
        Query the LLM with a prompt and return the response.
        """
        response = await self.async_client.chat.completions.create(
            model=self.ai_model_manager.DEFAULT_LLM_MODEL,
            messages=[{"role": "system", "content":prompts.extract_session_prompt_system()},
                      {"role": "user", "content": prompts.extract_session_prompt_user(query=query)}]
        )

        return response.choices[0].message.content
    
    # 기자 뉴스 모델 중 '최신' 혹은 '전체 기간' 대한 LLM을 호출하는 함수
    async def recent_or_global_cal_llm(self, query: str) -> str:
        """
        Query the LLM with a prompt and return the response.
        """
        response = await self.async_client.chat.completions.create(
            model=self.ai_model_manager.DEFAULT_LLM_MODEL,
            messages=[{"role": "user", "content": prompts.check_recent_or_global_prompt(query=query)}]
        )

        return response.choices[0].message.content

        
    def timer(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            print(f" | {func.__name__} 실행 시간: {end_time - start_time:.2f}초 | ")
            return result
        return wrapper

    @timer
    async def logical_routing(self, query: str) -> dict:
        """
        Route the query based on the intent identified by the LLM.
        """
        # print('start query routing')
        start_time = time.time()

        intent_response = await self.query_llm(query=query)

        # Parse LLM response (Basic parsing example)
        if "Time-Weighted Entity Retrieval" in intent_response:
            intent = "Time-Weighted Entity Retrieval"
        elif "General Q&A Retrieval" in intent_response:
            intent = "General Q&A Retrieval"
        elif "Session" in intent_response:
            intent = "Filtering-Based Session"
        elif "Date" in intent_response:
            intent = "Filtering-Based Date"
        elif "Time-Based News Summarization" in intent_response:
            intent = "Time-Based News Summarization"
        elif "Journalist" in intent_response:
            intent = "Journalist-Related Query"
        else:
            intent = "General Q&A Retrieval"

        # print('finish query routing')
        return {"intent": intent, "llm_response": intent_response}

    # 날짜를 계산하는 LLM 함수
    async def date_cal(self, query: str) -> dict:
        import json
        
        llm_output = await self.date_cal_llm(query=query)
        
        if "start_date" and "end_date" not in llm_output:
            print('No data: start_date or end_date')
            print('llm_output: ', llm_output)
            return []

        else:
            try:
                json_match = re.search(r"\[\s*\{.*?\}\s*\]", llm_output, re.DOTALL)
                if json_match:
                    cleaned_output = json_match.group()
                    outputs = json.loads(cleaned_output)
                else:
                    print("No valid JSON found in the output.")
                    return []
                # cleaned_output = llm_output.strip("```json\n").strip("\n```").strip()
                # cleaned_output = re.sub(r",\s*\]", "]", cleaned_output)
                # outputs = json.loads(cleaned_output)
                print(outputs)
                date_list = []
                for output in outputs:
                    start_date = datetime.strptime(output["start_date"], "%Y-%m-%d")
                    end_date = datetime.strptime(output["end_date"], "%Y-%m-%d")

                    # print('start date: ', start_date)
                    # print('end date: ', end_date)

                    current_date = start_date
                    while current_date <= end_date:
                        date_list.append(current_date.strftime("%Y-%m-%d"))
                        current_date += timedelta(days=1)

                return {"date_list": date_list, "date_range": outputs}
            
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                print('llm_output: ', llm_output)
                print(e)
                return []
            except Exception as e:
                print('llm_output: ', llm_output)
                print(e)
                return []

    # 총회 기간을 날짜로 바꾸는 파일과 함수
    with open('./app/assets/session_period.json', 'r') as file:
        session_period = json.load(file)

    def session_to_date_list(self, sessions):
        date_list = []
        for session in sessions:
            start_date = datetime.strptime(self.session_period[session - 1]["start_date"], "%Y-%m-%d")
            end_date = datetime.strptime(self.session_period[session - 1]["end_date"], "%Y-%m-%d")

            print(f"['start date': '{start_date}', 'end date': '{end_date}']")

            current_date = start_date
            while current_date <= end_date:
                date_list.append(current_date.strftime("%Y-%m-%d"))
                current_date += timedelta(days=1)

        return date_list

    async def extract_session_numbers(self, query: str) -> list:
        """
        Sends the query to the LLM and extracts session numbers.

        Args:
            query (str): User's input query.

        Returns:
            list: A list of session numbers or an error message.
        """

        llm_output = await self.session_cal_llm(query=query)

        try:
            # Try to parse the LLM output as JSON
            session_numbers = json.loads(llm_output)
            return session_numbers if isinstance(session_numbers, list) else []
        except json.JSONDecodeError:
            # If parsing fails, return the error message
            print(llm_output)
            print('no session number')
            return []

    # Query중 기자 '이름 추출 함수
    @staticmethod
    def extract_journalist_names(query: str):
        import re
        # 정규식: '기자' 또는 '기사' 앞의 단어를 추출하며 연결어 제거
        pattern = r'(\S+)\s*(?:기자|기자)|\b(\S+?)(?=와|과|및|,|\s(?:기자|기자))'
        # 정규식으로 매칭된 이름 추출
        matches = re.findall(pattern, query)
        # 정규식 결과 정리: 그룹에서 필요한 부분만 추출
        names = [name for match in matches for name in match if name]

        # 중복 제거 및 결과 반환
        if names:
            return list(set(names))
        else:
            print("No journalist mentioned in the query.")
            return []

    # Sources 리스트 추출 함수
    @staticmethod
    def extract_ids(answer_text:str):
        if "Sources:" in answer_text:
            return answer_text.split("Sources: [")[-1].rstrip("]").split(", ")
        return []
    
    def remove_duplicates_keep_order(self,lst):
        """
        Remove duplicates from a list while maintaining the original order.
        Only the first occurrence of each value is kept.

        Args:
        lst (list): The input list from which duplicates should be removed.

        Returns:
        list: A new list with duplicates removed and original order preserved.
        """
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                result.append(item)
                seen.add(item)
        return result

    def remove_hallucinated_sources(self, llm_text, id_list, sources_list):
        """
        Remove source markers from the LLM text that are not in the source list.

        Args:
        llm_text (str): The LLM response text.
        source_list (list): A list of valid source markers.

        Returns:
        str: The LLM text with hallucinated sources removed.
        """
        # Create a set of valid source markers
        valid_sources = set(id_list)

        # Find all source markers in the text
        all_sources = re.findall(r"\[\d+\]", llm_text)
        wrong_sources = []
        deleted_ids_list = []

        # Loop through all sources and remove hallucinated ones
        for source in set(all_sources):
            source_number = source.strip("[]")  # Extract the number without brackets
            if source_number not in valid_sources:
                print("Wrong source(hallucinated) in LLM response: ", source_number)
                wrong_sources.append(source_number)
                llm_text = llm_text.replace(source, "")  # Remove invalid source
                if source_number in id_list:
                    ind = id_list.index(source_number)
                    del sources_list[ind]
                    print('deleted source: ', id_list[ind])
                    deleted_ids_list.append(id_list[ind])
                    del id_list[ind]
        return {
            "updated_answer": llm_text,
            "sources": sources_list,
            "final_id_list": id_list,
            "wrong_sources": wrong_sources,
            "deleted_ids_list": deleted_ids_list
        }

    def replace_sources_with_indices(self, llm_text, source_list):
        """
        Replace source markers in the LLM response with indices based on the source list order.

        Args:
        llm_text (str): The original LLM response text.
        source_list (list): A list of source markers in the desired order.

        Returns:
        str: The LLM response with source markers replaced by their indices.
        """
        for index, source in enumerate(source_list, start=1):
            llm_text = llm_text.replace(f"[{source}]", f"[{index}]")
        return llm_text

    def remove_duplicate_references(self, text):
        def process_line(line):
            # 정규식으로 [숫자] 추출
            matches = re.findall(r"\[\d+\]", line)
            # 중복 제거 및 숫자 기준 정렬
            unique_numbers = sorted({int(match.strip('[]')) for match in matches})
            # 정렬된 [숫자] 형식으로 변환
            sorted_matches = ''.join(f"[{num}]" for num in unique_numbers)
            # 라인에서 [숫자] 제거 후 정렬된 [숫자] 추가
            line_without_refs = re.sub(r"\[\d+\]", "", line)
            return line_without_refs.strip() + ' ' + sorted_matches

        # 텍스트를 줄 단위로 나눠 처리
        lines = text.split('\n')
        processed_lines = [process_line(line) for line in lines]
        return '\n'.join(processed_lines)
    
    def makeing_source(self, result, id_list):
        # Extract sources list
        # print('makeing_source len(result): ', len(result['context']))
        print(f'origin id_list({len(id_list)})개: {id_list}')
        # Initialize the source list
        remove_duplicates_id_list = self.remove_duplicates_keep_order(id_list)
        print(f'remove_duplicates_id_list({len(remove_duplicates_id_list)})개: {remove_duplicates_id_list}')
        sources = [0] * len(remove_duplicates_id_list)
        check_id_list = remove_duplicates_id_list.copy()
        # Iterate over the context and populate the source list
        
        for ind, id in enumerate(remove_duplicates_id_list):
            for doc in result.get('context', []):
                # print(doc.id[:-2])
                if id == str(doc.id[:-2]):
                    # print('same id: ', id)
                    meta = doc.metadata
                    sources[ind] = {
                        "source": meta['source'],
                        "image_url": meta['images_url'],
                        "title": meta['title'],
                        "section": meta['primary_section'],
                        "date": f"{meta['init_date']} {meta['init_timestamp'][:-3]}",
                        "journalist_name": meta['journalist_name']
                    }
                    check_id_list[ind] = 'PASS'
                    # print(source[ind]) 
                    
        pass_id_list = [id for id, check in zip(remove_duplicates_id_list, check_id_list) if check == 'PASS']
        print(f'Check Halucinated ids[PASS({check_id_list.count("PASS")}개), FAIL({len(check_id_list)-check_id_list.count("PASS")}개)]: {check_id_list}')
        return {
            "sources": [source for source in sources if source != 0][:10],
            "pass_id_list": pass_id_list[:10],
            "id_list": id_list,
            "remove_duplicates_id_list": remove_duplicates_id_list,
            "check_id_list": check_id_list
        }


    # Sources 뒤를 제거하여 result의 Answer(답변)만 갖는 함수
    @timer
    async def get_answer(self,result):
        # "Sources: [...]" 패턴을 제거
        # clean_answer = re.sub(r"Sources: \[.*?\]", "", result['answer'], flags=re.DOTALL)
        # clean_answer = re.sub(r"Sources: \[.*?\]\s*\n?", "", result['answer'], flags=re.DOTALL)
        clean_answer = re.sub(r"Sources: \[.*?\]\s*(\n|$)", "", result['answer'], flags=re.DOTALL)
        id_list = self.extract_ids(result['answer'])
        making_sources = self.makeing_source(result, id_list)
        remove_hallucinated_sources = self.remove_hallucinated_sources(clean_answer.strip(), making_sources.get("pass_id_list"), making_sources.get("sources"))

        updated_answer = remove_hallucinated_sources.get("updated_answer")
        sources = remove_hallucinated_sources.get("sources")
        final_id_list = remove_hallucinated_sources.get("final_id_list")

        full_updated_answer = self.replace_sources_with_indices(updated_answer, final_id_list)
        final_answer = self.remove_duplicate_references(full_updated_answer)
        # 공백 정리
        print(f'answer: {final_answer[:30]}')
        return final_answer, sources, making_sources, remove_hallucinated_sources

    def _init_question_answer_chain(self):
        # prompt = ChatPromptTemplate.from_messages(prompts.custom_prompt_template())
        prompt = ChatPromptTemplate.from_messages(prompts.custom_prompt_template_id())
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        return question_answer_chain

    @timer
    async def timeweighted_LLM(self, query: str) -> dict:
        # 체인 실행
            # retriever에 직접 search_kwargs 설정
        # docs = await self.hybird_retriever.aget_relevant_documents(
        start_time = time.time()
        docs = await self.hybird_retriever.asimilarity_search(
            query,
            filter={
                    'section': {'$nin': ['기독AD']}, 
                    'init_year': {'$gte': datetime.now().year-2}
            }
        )
        document_end_time = time.time()
            # 검색된 문서로 chain 실행
        result = await self.question_answer_chain.ainvoke({
            "input": query,
            "context": docs,  # 검색된 문서 전달
            "current_time": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
            "MAX_TOKENS": self.ai_model_manager.DEFAULT_MAX_TOKEN
        })

        model_end_time = time.time()
        
        # result = self.timeweighted_rag_chain.invoke({
        #     "input": query,
        #     "current_time": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
        #     "MAX_TOKENS": self.ai_model_manager.DEFAULT_MAX_TOKEN
        # })
        return {
            "input": query,
            "answer": result,  # result가 dict 형태로 반환되므로
            "context": docs,
            "document_length": len(docs),
            "get_document_start_timestamp": start_time,
            "get_document_end_timestamp": document_end_time,
            "model_duration": model_end_time - start_time,
            "config": 0,
            "model_type": "TIME_WEIGHTED_LLM"
        }

    @timer
    async def hybird_dense_sparse_LLM(self, query: str) -> dict:
        
        start_time = time.time()
        # print('start_time: ', start_time)
        docs = await self.hybird_retriever.asimilarity_search(
            query,
            filter={
                'section': {'$nin': ['기독AD']}, 
            }
        )
        document_end_time = time.time()
        # print('end_time: ', end_time)
        # start_time = time.time()
        # docs = await self.hybird_retriever.ainvoke(
            # query,
            # search_kwargs={
            #     'filter': {
            #         'section': {'$nin': ['기독AD']}, 
            #     }
            # }
            # filter={
            #     'section': {'$nin': ['기독AD']}, 
            # }
        # )
        # end_time = time.time()
        # print(f" | {self.hybird_retriever._get_relevant_documents.__name__} 실행 시간: {end_time - start_time:.2f}초 | ")
        
        # 검색된 문서로 chain 실행
        result = await self.question_answer_chain.ainvoke({
            "input": query,
            "context": docs,  # 검색된 문서 전달
            "current_time": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
            "MAX_TOKENS": self.ai_model_manager.DEFAULT_MAX_TOKEN,
        })
        model_end_time = time.time()
        # result = await self.hybrid_rag_chain.ainvoke(
        #     {
        #         "input": query,
        #         "current_time": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
        #         "MAX_TOKENS": self.ai_model_manager.DEFAULT_MAX_TOKEN
        #     }
        # )

        return {
            "input": query,
            "answer": result,  # result가 dict 형태로 반환되므로
            "context": docs,
            "document_length": len(docs),
            "get_document_start_timestamp": start_time,
            "get_document_end_timestamp": document_end_time,
            "model_duration": model_end_time - start_time,
            "config": 0,
            "model_type": "GENERAL_QNA_LLM"
        }
        # return result

    @timer
    async def date_filter_LLM(self, query: str, date_cal: dict) -> dict:
        date_list = date_cal.get("date_list")
        date_filtering_vectorstore = self._init_date_filter_score_retriever(date_list)
        start_time = time.time()
        # print('start_time: ', start_time)
        docs = await date_filtering_vectorstore._aget_relevant_documents(query+' 총회')
        document_end_time = time.time()
        # print('end_time: ', end_time)
        # rag_chain = create_retrieval_chain(date_filtering_vectorstore, self.question_answer_chain)

        # result = await rag_chain.ainvoke(
        result = await self.question_answer_chain.ainvoke(
            {
                # "input": query + ' 총회',
                "input": query,
                "context": docs,
                "current_time": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
                "MAX_TOKENS": self.ai_model_manager.DEFAULT_MAX_TOKEN
            }
        )
        model_end_time = time.time()

        return {
            "input": query,
            "answer": result,  # result가 dict 형태로 반환되므로
            "context": docs,
            "document_length": len(docs),
            "get_document_start_timestamp": start_time,
            "get_document_end_timestamp": document_end_time,
            "model_duration": model_end_time - start_time,
            "config": date_cal.get("date_range"),
            "model_type": "DATE_FILTER_LLM"
        }

    @timer
    async def summary_filter_LLM(self, query: str, date_cal: dict) -> dict:
        date_list = date_cal.get("date_list")
        filtering_vectorstore = self._init_summary_filter_retriever(date_list)

        start_time = time.time()
        # print('start_time: ', start_time)
        relevant_docs = await filtering_vectorstore._aget_relevant_documents("query")
        print('len(relevant_docs): ', len(relevant_docs))
        # relevant_docs = await filtering_vectorstore._aget_relevant_documents(
        #     query,
        #     search_kwargs={
        #         'k': 50,
        #         "filter": {
        #             "init_date": {"$in": date_list},
        #             "section": {"$nin": ['설교', '기독AD', '오피니언']}
        #         },
        #         'setting': "summary and no_query_embedding"
        #     })
        document_end_time = time.time()
        # print('end_time: ', end_time)

        # 요약 작업 수행
        answer = await self.summary_chain.ainvoke({
            "input": query, 
            "MAX_TOKENS": self.ai_model_manager.DEFAULT_MAX_TOKEN, 
            "context": relevant_docs
        })
        model_end_time = time.time()

        return {
            "input": query,
            "answer": answer,
            "context": relevant_docs,
            "document_length": len(relevant_docs),
            "get_document_start_timestamp": start_time,
            "get_document_end_timestamp": document_end_time,
            "model_duration": model_end_time - start_time,
            "config": date_cal.get("date_range"),
            "model_type": "SUMMARY_FILTER_LLM"
        }

    @timer
    async def journalist_filter_LLM(self, query: str, name_list: list) -> dict:
        check = await self.recent_or_global_cal_llm(query)
        if 'recent' in check:
            type = 'recent'
        else:
            type = 'global'
        print('type: ', type)
        
        
        jounaralist_time_filtering_retriever = self._init_jounaralist_time_filter_retriever(name_list, type)
        start_time = time.time()
        # print('start_time: ', start_time)
        docs = await jounaralist_time_filtering_retriever._aget_relevant_documents(query)
        document_end_time = time.time()
        # print('end_time: ', end_time)
        # rag_chain = create_retrieval_chain(jounaralist_time_filtering_retriever, self.journalist_chain)
        
        # result = await rag_chain.ainvoke(
        result = await self.journalist_chain.ainvoke(
            {
                "input": query,
                "context": docs,
                "current_time": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
                "MAX_TOKENS": self.ai_model_manager.DEFAULT_MAX_TOKEN
            }
        )
        model_end_time = time.time()

        # return result
        return {
            "input": query,
            "answer": result,  # result가 dict 형태로 반환되므로
            "context": docs,
            "document_length": len(docs),
            "get_document_start_timestamp": start_time,
            "get_document_end_timestamp": document_end_time,
            "model_duration": model_end_time - start_time,
            "config": name_list,
            "model_type": type.upper() + "_" + "JOURNAL_LIST_FILTER_LLM"
        }

    
    @timer
    async def query_model_pipeline(self, query: str):
        print('*'*60)
        print('Search start time:',datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("query:",query)
        routing = {}
        routing = await self.logical_routing(query)
        intent = routing.get('intent')

        if "Time-Weighted Entity Retrieval" in intent:
            print("----------- MODLE: TIME-WEIGHTED -----------")
            return await self.timeweighted_LLM(query)

        elif "General Q&A Retrieval" in intent:
            print("----------- MODLE: GENERAL Q&A -----------")
            return await self.hybird_dense_sparse_LLM(query)

        elif "Session" in intent:
            print("----------- MODLE: SESSION -----------")
            sessions = await self.extract_session_numbers(query)
            if not sessions:
                print("We can't get sessions. so trun to general Q&A")
                return await self.hybird_dense_sparse_LLM(query)
            date_list = self.session_to_date_list(sessions)
            date_cal = {
                "date_list": date_list,
                "date_range": sessions,
            }

            return await self.date_filter_LLM(query, date_cal)

        elif "Date" in intent:
            print("----------- MODLE: DATE FILTERING -----------")
            date_cal = await self.date_cal(query)

            if not date_cal.get("date_list"):
                print('date_list: ', date_cal.get("date_list"))
                print("We can't get date_list. so trun to general Q&A")
                return await self.hybird_dense_sparse_LLM(query)

            return await self.date_filter_LLM(query, date_cal)

        elif "Time-Based News Summarization" in intent:
            print("----------- MODLE: NEWS SUMMARIZATION -----------")
            date_list = await self.date_cal(query)
            if not date_list:
                print('date_list: ', date_list)
                print("We can't get date_list. so trun to general Q&A")
                return await self.hybird_dense_sparse_LLM(query)

            return await self.summary_filter_LLM(query, date_list)

        elif "Journalist-Related Query" in intent:
            print("----------- MODLE: JOURNALIST-RELATED QUERY -----------")
            name_list = self.extract_journalist_names(query)
            if not name_list:
                print('name_list: ', name_list)
                print("We can't get name_list. So turn to genernal Q&A")
                return await self.hybird_dense_sparse_LLM(query)
            print(f'Journalist name list: {name_list}')
            return await self.journalist_filter_LLM(query, name_list)




















    def query(self, query: str):
        rag_result = self.rag_chain.invoke(
            {
                "input": query,
                "current_time": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
                "MAX_TOKENS": self.ai_model_manager.DEFAULT_MAX_TOKEN
            }
        )

        source_set = set()
        for document in rag_result.get("context"):
            meta = document.metadata
            source_set.add(
                f"source: {meta['source']}, title: {meta['title']}  {meta['mod_date']} {meta['mod_timestamp']}"
            )
        response = {
            "rag_result": rag_result.get("answer", "No answer generated"),
            "sources": source_set
        }
        return response

    def get_documents(self, query: str, k=5) -> List[Document]:
        docs = self.pinecone_retriever.invoke(query, k=k)
        return docs

    @staticmethod
    def get_source(docs: List[Document]) -> List[dict]:
        source_list = []
        for document in docs:
            meta = document.metadata
            new_item = {
                    "source": meta["source"],
                    "title": meta["title"],
                    "date": f"{meta['init_date']} {meta['init_timestamp']}",
                    "image_url": meta["images_url"],
                }
            if new_item not in source_list:
                source_list.append(new_item)
        return source_list

    async def stream_query(self, query: str, docs: List[Document]) -> AsyncGenerator[str, None]:
        answer = ""
        try:
            async for event in self.question_answer_chain.astream(
                    {
                        "input": query,
                        'context': docs,
                        "current_time": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
                        "MAX_TOKENS": self.ai_model_manager.DEFAULT_MAX_TOKEN
                    }
            ):
                answer += event
                yield event
            await put_search_response_tracking(query=query, answer=answer)

        except Exception as e:
            yield f"Error: {str(e)}\n"


rag_pipeline = RagPipeline()
