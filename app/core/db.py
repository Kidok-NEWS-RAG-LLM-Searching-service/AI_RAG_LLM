import certifi
from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import settings


class MongoDB:
    def __init__(self):
        self.client = None

    def connect(self):
        self.client = AsyncIOMotorClient(settings.mongo_db_url,tlsCAFile=certifi.where())
        self.db = self.client[settings.mongo_db_name]
        self.cache = self.client[settings.mongo_cache_name]
        self.dashboard_db = self.client[settings.mongo_dashboard_db_name]

    def close(self):
        self.client.close()

    def get_db(self):
        return self.db

    def get_cache(self):
        return self.cache

    def get_dashboard_db(self):
        return self.dashboard_db


mongodb = MongoDB()
mongodb.connect()
