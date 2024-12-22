import os
from datetime import datetime, timedelta
from typing import List
import json
import re

import pandas as pd
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app.api.service.encoders.encoders import sparse_encoder
from app.api.service.logs.log import put_search_response_tracking
from app.api.service.managers.stop_words_manager import StopwordsManager
from app.api.service.retrievers.PineconeKiwiHybridRetriever import PineconeKiwiHybridRetriever
from app.api.service.retrievers.TimeWeightedCustomVectorStoreRetriever import TimeWeightedCustomVectorStoreRetriever
from app.api.service.retrievers.TimeWeightedJounaralistFilteringVectorStoreRetriever import TimeWeightedJounaralistFilteringVectorStoreRetriever
from app.core.llm import AIModelManager
from app.core.pinecone_index_initializer import PineconeIndexInitializer
from app.core import prompts
from app.core.init_method import InitVectorStore
from app.core.vectorstore import CustomPineconeVectorStore

from typing import AsyncGenerator


current_dir = os.path.dirname(os.path.abspath(__file__))


class RagPipeline:
    stop_words_manager = StopwordsManager()
    ai_model_manager = AIModelManager()

    llm = ai_model_manager.llm
    embeddings = ai_model_manager.embeddings
    client = ai_model_manager.client

    # sparse_encoder_path = os.path.join("./app/news_rag_llm/yong_contextual_sparse_encoder.pkl")
    sparse_encoder_path = os.path.join("./app/sparse_encoder_folder/sparse_encoder_1_57000.pkl")
    global_source_set = set()

    if not os.path.exists(sparse_encoder_path):
        print(f"{sparse_encoder_path} not found. Creating sparse encoder...")
        contextual_chunks_df = pd.read_parquet("./app/assets/contents_1_80000.parquet", engine="pyarrow")
        sparse_encoder_value = sparse_encoder.create_sparse_encoder(
            stop_words_manager.fetch_stopwords(), mode="kiwi"
        )
        saved_path = sparse_encoder.fit(
            bm25_encoder=sparse_encoder_value,
            contents=contextual_chunks_df.contexts.tolist(),
            save_path=sparse_encoder_path
        )
        print(f"Sparse encoder saved at: {saved_path}")

    pinecone_index_initializer = PineconeIndexInitializer(
        # sparse_encoder_path="./app/news_rag_llm/yong_contextual_sparse_encoder.pkl",
        sparse_encoder_path="./app/sparse_encoder_folder/sparse_encoder_1_57000.pkl",
        stopwords=stop_words_manager.fetch_stopwords(),  # 불용어 사전
        tokenizer="kiwi",
        embeddings=embeddings,
        top_k=20,
        alpha=0.3,
    )

    init_data = pinecone_index_initializer.get_pinecone_init_data()

    pinecone_retriever = PineconeKiwiHybridRetriever(
        embeddings=init_data["embeddings"],
        sparse_encoder=init_data["sparse_encoder"],
        index=init_data["index"],
        top_k=init_data["top_k"],
        alpha=init_data["alpha"],
        namespace=init_data["namespace"]
    )

    init_vectorstore = InitVectorStore()
    vectorstore = init_vectorstore.init_pinecone_vectorstore()
    add_jounaralist_name_customize_vectorstore = init_vectorstore.init_customize_vectorstore()

    custom_vectorstore = CustomPineconeVectorStore(base_store=vectorstore)
    jounaralist_customize_vectorstore = CustomPineconeVectorStore(base_store=add_jounaralist_name_customize_vectorstore)

    def _init_timeweighted_retriever(self):
        # Retriever 초기화
        return TimeWeightedCustomVectorStoreRetriever(
            vectorstore=self.custom_vectorstore,
            decay_rate=0.000_1,  # 0.000_000_1
            k=20,  # 반환할 최대 문서 개수
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.319, }
        )

    def _init_date_filter_score_retriever(self, date_list: list):
        return self.custom_vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                'score_threshold': 0.319,  # 원래 0.67이였음(similarity_search_with_relevance_scores 계산 방식으로 정규화가 되기 때문에)
                # 그런데 내가 해당 함수 cumstomize하면서 그냥 정규화 안된 score로 점수 거르게 만들어서 이렇게 점수 사용.
                'k': 20,
                "filter": {
                    "init_date": {"$in": date_list},
                    "section": {"$nin": ['설교', '기독AD', '오피니언']}
                }
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
                'setting': 'summary'
            }
        )

    def _init_jounaralist_time_filter_retriever(self, name_list: list):
        return TimeWeightedJounaralistFilteringVectorStoreRetriever(
            vectorstore=self.jounaralist_customize_vectorstore,
            decay_rate=0.000_1,  # 0.000_000_1
            k=20,  # 반환할 최대 문서 개수
            name_list=name_list
        )

    init_data = pinecone_index_initializer.get_pinecone_init_data()
    hybird_retriever = PineconeKiwiHybridRetriever(
        embeddings=init_data["embeddings"],
        sparse_encoder=init_data["sparse_encoder"],
        index=init_data["index"],
        top_k=init_data["top_k"],
        alpha=init_data["alpha"],
        namespace=init_data["namespace"]
    )
    timeweighted_retriever = _init_timeweighted_retriever()



    prompt = ChatPromptTemplate.from_template(AIModelManager.get_custom_prompt_template_v2())
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(pinecone_retriever, question_answer_chain)

    # query routing을 위한 LLM 호출하는 함수
    def query_llm(self, query: str, model: str = ai_model_manager.DEFAULT_LLM_MODEL) -> str:
        """
        Query the LLM with a prompt and return the response.
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompts.query_routing_prompt_system()},
                      {"role": "user", "content": prompts.query_routing_prompt_user(query=query)}]
        )

        return response.choices[0].message.content

    # 날짜 계산에 대한 LLM을 호출하는 함수
    def date_cal_llm(self, query: str, model: str = ai_model_manager.DEFAULT_LLM_MODEL) -> str:
        """
        Query the LLM with a prompt and return the response.
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompts.date_cal_prompt_system()},
                      {"role": "user", "content": prompts.date_cal_prompt_user(query=query)}]
        )

        return response.choices[0].message.content

    # 날짜 계산에 대한 LLM을 호출하는 함수
    def session_cal_llm(self, query: str, model: str = ai_model_manager.DEFAULT_LLM_MODEL) -> str:
        """
        Query the LLM with a prompt and return the response.
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content":prompts.extract_session_prompt_system()},
                      {"role": "user", "content": prompts.extract_session_prompt_user(query=query)}]
        )

        return response.choices[0].message.content

    def logical_routing(self, query: str) -> dict:
        """
        Route the query based on the intent identified by the LLM.
        """
        print('start query routing')

        intent_response = self.query_llm(query=query)

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

        print('finish query routing')
        return {"intent": intent, "llm_response": intent_response}

    # 날짜를 계산하는 LLM 함수
    def date_cal(self, query: str) -> list:
        llm_output = self.date_cal_llm(query=query)
        if "start_date" and "end_date" not in llm_output:
            print('No data: start_date or end_date')
            return []

        else:
            try:
                import json
                cleaned_output = llm_output.strip("```json\n").strip("\n```").strip()
                outputs = json.loads(cleaned_output)
                print(outputs)
                date_list = []
                for output in outputs:
                    start_date = datetime.strptime(output["start_date"], "%Y-%m-%d")
                    end_date = datetime.strptime(output["end_date"], "%Y-%m-%d")

                    print('start date: ', start_date)
                    print('end date: ', end_date)

                    current_date = start_date
                    while current_date <= end_date:
                        date_list.append(current_date.strftime("%Y-%m-%d"))
                        current_date += timedelta(days=1)

                return date_list
            except Exception as e:
                print('llm_output: ', llm_output)
                print(e)
                return []

    # 총회 기간을 날짜로 바꾸는 파일과 함수
    with open('../../assets/session_period.json', 'r') as file:
        session_period = json.load(file)

    def session_to_date_list(self, sessions):
        date_list = []
        for session in sessions:
            start_date = datetime.strptime(self.session_period[session - 1]["start_date"], "%Y-%m-%d")
            end_date = datetime.strptime(self.session_period[session - 1]["end_date"], "%Y-%m-%d")

            print('start date: ', start_date)
            print('end date: ', end_date)

            current_date = start_date
            while current_date <= end_date:
                date_list.append(current_date.strftime("%Y-%m-%d"))
                current_date += timedelta(days=1)

        return date_list

    def extract_session_numbers(self, query: str) -> list:
        """
        Sends the query to the LLM and extracts session numbers.

        Args:
            query (str): User's input query.

        Returns:
            list: A list of session numbers or an error message.
        """

        llm_output = self.session_cal_llm(query=query)

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
    def extract_sources(answer_text:str):
        if "Sources:" in answer_text:
            return answer_text.split("Sources: [")[-1].rstrip("]").split(", ")
        return []

    # LLM 최종 답변 중 source 추출 함수
    def makeing_source(self, result):
        # Extract sources list
        sources_list = self.extract_sources(result['answer'])
        print(sources_list)
        # Create a mapping for sources to their index
        sources_index = {doc_id: idx for idx, doc_id in enumerate(sources_list)}
        # Initialize the source list
        source = [0] * len(sources_list)
        # Iterate over the context and populate the source list
        for doc in result.get('context', []):
            if doc.id[:-2] in sources_index:
                # print(doc.id)
                idx = sources_index[doc.id[:-2]]
                meta = doc.metadata
                source[idx] = (
                    f"source: {meta['source']}, title: {meta['title']}, "
                    f"section: {meta['primary_section']}, {meta['init_date']} "
                    f"{meta['init_timestamp']} {meta['journalist_name']}"
                )
        return source

    # Sources 뒤를 제거하여 result의 Answer(답변)만 갖는 함수
    @staticmethod
    def get_answer(result):
        # "Sources: [...]" 패턴을 제거
        clean_answer = re.sub(r"Sources: \[.*?\]", "", result['answer'], flags=re.DOTALL)
        # 공백 정리
        return clean_answer.strip()

    def _init_question_answer_chain(self):
        prompt = ChatPromptTemplate.from_messages(prompts.custom_prompt_template())
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        return question_answer_chain

    def timeweighted_LLM(self, query: str) -> dict:
        question_answer_chain = self._init_question_answer_chain()
        rag_chain = create_retrieval_chain(self.timeweighted_retriever, question_answer_chain)

        # 체인 실행
        result = rag_chain.invoke({
            "input": query,
            "current_time": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
            "MAX_TOKENS": self.ai_model_manager.DEFAULT_MAX_TOKEN
        })

        return result

    def hybird_dense_sparse_LLM(self, query: str) -> dict:
        # params 초기설정
        question_answer_chain = self._init_question_answer_chain()
        rag_chain = create_retrieval_chain(self.hybird_retriever, question_answer_chain)

        result = rag_chain.invoke(
            {
                "input": query,
                "current_time": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
                "MAX_TOKENS": self.ai_model_manager.DEFAULT_MAX_TOKEN
            }
        )

        return result

    def date_filter_LLM(self, query: str, date_list: list) -> dict:
        date_filtering_vectorstore = self._init_date_filter_score_retriever(date_list)
        question_answer_chain = self._init_question_answer_chain()
        rag_chain = create_retrieval_chain(date_filtering_vectorstore, question_answer_chain)

        result = rag_chain.invoke(
            {
                "input": query + '총회',
                "current_time": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
                "MAX_TOKENS": self.ai_model_manager.DEFAULT_MAX_TOKEN
            }
        )

        return result

    def summary_filter_LLM(self, query: str, date_list: list) -> dict:
        summary_prompt = prompts.summary_prompt_template()
        filtering_vectorstore = self._init_summary_filter_retriever(date_list)

        # prompt = ChatPromptTemplate.from_template(summary_prompt)
        prompt = ChatPromptTemplate.from_messages(summary_prompt)
        summary_chain = create_stuff_documents_chain(self.llm, prompt)
        relevant_docs = filtering_vectorstore._get_relevant_documents(" ")

        # 요약 작업 수행
        answer = summary_chain.invoke({"input": query, "MAX_TOKENS": self.ai_model_manager.DEFAULT_MAX_TOKEN, "context": relevant_docs})
        result = {}
        result['input'] = query
        result['answer'] = answer
        result['context'] = relevant_docs
        return result

    def journalist_filter_LLM(self, query: str, name_list: list) -> dict:
        jounaralist_prompt = ChatPromptTemplate.from_messages(prompts.jounarlist_prompt_template())
        jounaralist_time_filtering_retriever = self._init_jounaralist_time_filter_retriever(name_list)

        question_answer_chain = create_stuff_documents_chain(self.llm, jounaralist_prompt)
        rag_chain = create_retrieval_chain(jounaralist_time_filtering_retriever, question_answer_chain)

        result = rag_chain.invoke(
            {
                "input": query,
                "current_time": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
                "MAX_TOKENS": self.ai_model_manager.DEFAULT_MAX_TOKEN
            }
        )

        return result

    def query_model_pipeline(self, query: str):

        routing = {}
        routing = self.logical_routing(query)
        intent = routing.get('intent')

        if "Time-Weighted Entity Retrieval" in intent:
            print("time-weighted")
            return self.timeweighted_LLM(query)

        elif "General Q&A Retrieval" in intent:
            print("General Q&A")
            return self.hybird_dense_sparse_LLM(query)

        elif "Session" in intent:
            print("session filtering")
            sessions = self.extract_session_numbers(query)
            if not sessions:
                print("We can't get sessions. so trun to general Q&A")
                return self.hybird_dense_sparse_LLM(query)

            print(sessions)
            return self.date_filter_LLM(query, self.session_to_date_list(sessions))

        elif "Date" in intent:
            print("Date filtering")
            date_list = self.date_cal(query)
            if not date_list:
                print('date_list: ', date_list)
                print("We can't get date_list. so trun to general Q&A")
                return self.hybird_dense_sparse_LLM(query)

            return self.date_filter_LLM(query, date_list)

        elif "Time-Based News Summarization" in intent:
            print("News Summarization")
            date_list = self.date_cal(query)
            if not date_list:
                print('date_list: ', date_list)
                print("We can't get date_list. so trun to general Q&A")
                return self.hybird_dense_sparse_LLM(query)

            return self.summary_filter_LLM(query, date_list)

        elif "Journalist-Related Query" in intent:
            print("Journalist-Related Query")
            name_list = self.extract_journalist_names(query)
            if not name_list:
                print('name_list: ', name_list)
                print("We can't get name_list. So turn to genernal Q&A")
                return self.hybird_dense_sparse_LLM(query)
            print(name_list)
            return self.journalist_filter_LLM(query, name_list)










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
