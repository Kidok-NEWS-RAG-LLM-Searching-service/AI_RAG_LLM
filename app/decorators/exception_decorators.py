import logging
from typing import Any

from fastapi import Request

logger = logging.getLogger(__name__)


def exception_log(handler_func):
    async def wrapper(request: Request, exc: Any):
        logger.error(f"{exc.code} occurred: {exc.detail}")
        return await handler_func(request, exc)
    return wrapper

