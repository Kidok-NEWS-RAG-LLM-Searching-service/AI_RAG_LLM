import time
from uuid import uuid4

from boto3.resources.base import ServiceResource

from app.api.repository.web_log_type import WebLogType
from app.core.db import mongodb


class WebLogRepository:

    def __init__(self, db: ServiceResource) -> None:
        self.__db = db

    async def put_item(
            self,
            log_type: WebLogType,
            redirect_url: str
    ):
        web_log = {
            "id": str(uuid4()),
            "type": log_type,
            "redirect_url": redirect_url,
            "timestamp": int(time.time())
        }
        await self.__db["web_log"].insert_one(web_log)


web_log_repository = WebLogRepository(mongodb.get_db())
