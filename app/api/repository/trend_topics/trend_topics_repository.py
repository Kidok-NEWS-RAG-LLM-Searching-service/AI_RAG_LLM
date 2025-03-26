from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.db import mongodb


class TrendTopicRepository:

    def __init__(self, db: AsyncIOMotorDatabase):
        self.__collection = db["trend_topic"]

    async def find_today_trend_topics(self, today_date: str):
        cursor = self.__collection.find({"date": today_date}).sort([
            ("date", 1),
            ("order", 1)
        ])
        trend_topics_in_collection = await cursor.to_list(length=None)
        print(len(trend_topics_in_collection))
        trend_topics = []

        for trend_topic in trend_topics_in_collection:
            trend_topics.append({
                "id": trend_topic["id"],
                "topic": trend_topic["topic"],
                "date": trend_topic["date"],
                "order": trend_topic["order"]
            })
        return trend_topics


trend_topic_repository = TrendTopicRepository(mongodb.get_dashboard_db())
