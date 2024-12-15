from app.api.repository.ai_search_log_repository import ai_search_log_repository
from app.api.repository.ai_search_log_type import AISearchLogType


async def put_search_tracking(query: str):
    await ai_search_log_repository.put_item(
        log_type=AISearchLogType.SEARCH,
        query=query,
    )


async def put_search_response_tracking(query: str, answer: str):
    await ai_search_log_repository.put_item(
        log_type=AISearchLogType.ANSWER,
        query=query,
        answer=answer
    )


async def put_search_response_like_tracking(query: str, answer: str, like: int):
    await ai_search_log_repository.put_item(
        log_type=AISearchLogType.LIKE_BUTTON_ONLY,
        query=query,
        answer=answer,
        like=like
    )


async def put_search_response_like_with_feedback_tracking(query: str, answer: str, like: int, feedback: str):
    await ai_search_log_repository.put_item(
        log_type=AISearchLogType.DESCRIPTION,
        query=query,
        answer=answer,
        like=like,
        description=feedback
    )

