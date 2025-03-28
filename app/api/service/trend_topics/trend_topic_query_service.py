from datetime import date

from app.api.repository.trend_topics.trend_topics_repository import trend_topic_repository


class TrendTopicQueryService:

    def __init__(self):
        self.trend_topic_repository = trend_topic_repository

    async def find_today_trend_topics(self):
        # today = date.today().isoformat()
        today = date.today()
        print(today)
        trend_topics = await self.trend_topic_repository.find_today_trend_topics(today_date=today)
        return trend_topics


trend_topic_query_service = TrendTopicQueryService()
