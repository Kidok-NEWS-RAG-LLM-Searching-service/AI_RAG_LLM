import time
from uuid import uuid4

from app.api.repository.web_log_type import WebLogType


class WebLogRepository:
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