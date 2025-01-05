from fastapi import APIRouter
from pydantic import BaseModel

from app.api.repository.web_log_repository import web_log_repository
from app.api.repository.web_log_type import WebLogType
from app.api.service.logs.log import put_search_tracking, put_search_response_tracking, \
    put_search_response_like_tracking, put_search_response_like_with_feedback_tracking

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


class QueryAnswerRequest(BaseModel):
    query: str
    answer: str


class QueryAnswerLikeRequest(BaseModel):
    query: str
    answer: str
    like: int


class QueryAnswerLikeWithFeedbackRequest(BaseModel):
    query: str
    answer: str
    like: int
    feedback: str


class TrackingWithRedirectUrlRequest(BaseModel):
    url: str


@router.post("/search")
async def tracking_query(request: QueryRequest):
    await put_search_tracking(query=request.query)
    return "success"


@router.post("/search/response")
async def tracking_query(request: QueryAnswerRequest):
    await put_search_response_tracking(query=request.query, answer=request.answer)
    return "success"


@router.post("/search/response/like")
async def tracking_query(request: QueryAnswerLikeRequest):
    await put_search_response_like_tracking(query=request.query, answer=request.answer, like=request.like)
    return "success"


@router.post("/search/response/feedback")
async def tracking_query(request: QueryAnswerLikeWithFeedbackRequest):
    await put_search_response_like_with_feedback_tracking(query=request.query, answer=request.answer, like=request.like, feedback=request.feedback)
    return "success"


@router.post("/web/home")
async def tracking_redirect_kidok_service():
    await web_log_repository.put_item(
        WebLogType.AI_SEARCH_SERVICE_TO_MAIN_KIDOK_SERVICE,
        "https://www.kidok.com"
    )
    return "success"


@router.post("/web/ai-result/sources")
async def tracking_ai_result_sources_redirect_kidok_service(request: TrackingWithRedirectUrlRequest):
    await web_log_repository.put_item(
        WebLogType.AI_SEARCH_RESULT_SOURCES_TO_MAIN_KIDOK_SERVICE,
        request.url
    )
    return "success"
