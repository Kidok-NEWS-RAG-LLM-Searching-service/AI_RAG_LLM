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
            log_type: AIModelPerformanceType,
            query: str,
            query_routing_start_timestamp: int,
            query_routing_end_timestamp: int,
    ):
        ai_search_log = {
            "id": str(uuid4()),
            "type": log_type,
            "query_routing_start_timestamp": int(time.time()),
            "query_routing_end_timestamp": int(time.time()),
            "query_routing_duration": query_routing_end_timestamp - query_routing_start_timestamp
        }
        await self.__db["ai_model_performance"].insert_one(ai_search_log)


ai_model_performance_log_repository = AIModelPerformanceLogRepository(mongodb.get_db())
