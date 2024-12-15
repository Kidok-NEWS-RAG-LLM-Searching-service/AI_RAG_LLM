import time
from uuid import uuid4

from boto3.resources.base import ServiceResource

from app.api.repository.ai_search_log_type import AISearchLogType
from app.core.db import mongodb


class AISearchLogRepository:
    def __init__(self, db: ServiceResource) -> None:
        self.__db = db

    async def put_item(
            self,
            log_type: AISearchLogType,
            query: str = "",
            answer: str = "",
            like: int = 0,
            description: str = ""
    ):
        ai_search_log = {
            "id": str(uuid4()),
            "type": log_type,
            "query": query,
            "answer": answer,
            "like": like,
            "description": description,
            "timestamp": int(time.time())
        }
        await self.__db["ai_search_log"].insert_one(ai_search_log)


ai_search_log_repository = AISearchLogRepository(mongodb.get_db())