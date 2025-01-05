import time
from uuid import uuid4

from boto3.resources.base import ServiceResource
from datetime import datetime, timezone
from app.core.db import mongodb


class CacheRepository:
    def __init__(self, cache: ServiceResource) -> None:
        self.__cache = cache

    async def put_item(
        self,
        answer_model_type: str,
        intent_model_type: str,
        query: str,
        query_embed: str,
        answer: str,
        config: any,
        get_document_start_timestamp: int,
        get_document_end_timestamp: int,
        len_document: int,
        model_duration: int,
        id_list: list,
        remove_duplicates_id_list: list,
        check_id_list: list,
        wrong_sources: list,
        deleted_ids_list: list,
        sources_list: list
    ):
        check_id_list = check_id_list or []
        web_log = {
            "id": str(uuid4()),
            "answer_model_type": answer_model_type,
            "intent_model_type": intent_model_type,
            "query": query,
            "query_embed": query_embed,  # Query vector data
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
            "sources_list": sources_list,
            "timestamp": int(time.time())
        }
        await self.__cache["ai_response_cache_store"].insert_one(web_log)

    async def get_today_cache_data(self):
        today = datetime.now(timezone.utc).date()
        start_of_today = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)
        end_of_today = datetime.combine(today, datetime.max.time(), tzinfo=timezone.utc)
        start_timestamp = int(start_of_today.timestamp())
        end_timestamp = int(end_of_today.timestamp())

        data = await self.__cache["ai_response_cache_store"].find({
            "timestamp": {"$gt": start_timestamp, "$lt": end_timestamp}
        }).to_list(length=None)

        return data


cache_repository = CacheRepository(mongodb.get_cache())
