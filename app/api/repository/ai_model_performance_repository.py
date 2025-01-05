import time
from uuid import uuid4

from boto3.resources.base import ServiceResource

from app.api.repository.ai_model_performance_type import AIModelPerformanceType
from app.core.db import mongodb


class AIModelPerformanceLogRepository:
    def __init__(self, db: ServiceResource) -> None:
        self.__db = db

    async def put_item(
            self,
            model_type: str,
            query: str,
            answer: str,
            config: any,
            get_document_start_timestamp: int,
            get_document_end_timestamp: int,
            len_document: int,
            model_duration: int,
            query_routing_start_timestamp: int,
            query_routing_end_timestamp: int,
            id_list: list,
            remove_duplicates_id_list: list,
            check_id_list: list,
            wrong_sources: list,
            deleted_ids_list: list
    ):
        ai_search_log = {
            "id": str(uuid4()),
            "query_routing_start_timestamp": query_routing_start_timestamp,
            "model_type": model_type,
            "query": query,
            "answer": answer,
            "config": config,
            "get_document_duration": get_document_end_timestamp - get_document_start_timestamp,
            "len_document": len_document,
            "model_duration": model_duration,
            "origin_id_list": id_list,
            "len_origin_id_list": len(id_list),
            "remove_duplicates_id_list": remove_duplicates_id_list,
            "len_remove_duplicates_id_list": len(remove_duplicates_id_list),
            "hallucination_check": check_id_list,
            "len_hallucination_check_pass": check_id_list.count("PASS"),
            "hallucination_in_llm_response": wrong_sources,
            "deleted_sources_in_llm": deleted_ids_list,
            "query_routing_duration": query_routing_end_timestamp - query_routing_start_timestamp,
        }
        await self.__db["ai_model_performance"].insert_one(ai_search_log)


ai_model_performance_log_repository = AIModelPerformanceLogRepository(mongodb.get_db())
