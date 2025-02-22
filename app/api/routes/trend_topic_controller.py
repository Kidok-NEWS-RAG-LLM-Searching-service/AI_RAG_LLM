from fastapi import APIRouter

from app.api.service.trend_topics.trend_topic_query_service import trend_topic_query_service

router = APIRouter()


@router.get("")
async def get_trend_topics():
    response = await trend_topic_query_service.find_today_trend_topics()
    return response
