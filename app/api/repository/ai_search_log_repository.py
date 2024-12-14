import time
from typing import Optional
from uuid import uuid4

from boto3.dynamodb.conditions import Attr
from boto3.resources.base import ServiceResource

from app.api.repository.ai_search_log_type import AISearchLogType
from app.core.config import settings
from app.core.db import log_db


class AISearchLogRepository:
    def __init__(self, db: ServiceResource) -> None:
        self.__db = db
        self.__table = self.__db.Table(settings.ai_search_log_table)

    def get_all(self):
        response = self.__table.scan()
        items = response.get("Items")
        return items

    def get_by_column(self, column_name: str, value: str, limit: Optional[int] = None, sort: Optional[str] = None):
        """
        Retrieve items filtered by a specific column with optional limit and sort.
        """
        # Use scan with filter expression
        response = self.__table.scan(
            FilterExpression=Attr(column_name).eq(value)
        )
        items = response.get("Items", [])

        # Apply sorting if specified
        if sort:
            reverse = sort.lower() == "desc"
            items = sorted(items, key=lambda x: x.get("timestamp", 0), reverse=reverse)

        # Apply limit if specified
        if limit:
            items = items[:limit]

        return items

    def put_item(
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
        self.__table.put_item(Item=ai_search_log)


ai_search_log_repository = AISearchLogRepository(log_db)