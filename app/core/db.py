import boto3
import certifi
from boto3.resources.base import ServiceResource
from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import settings


class MongoDB:
    def __init__(self):
        self.client = None

    def connect(self):
        self.client = AsyncIOMotorClient(settings.mongo_db_url,tlsCAFile=certifi.where())
        self.db = self.client[settings.mongo_db_name]
        print("DB 와 연결되었습니다.")

    def close(self):
        self.client.close()

    def get_db(self):
        return self.db


mongodb = MongoDB()
mongodb.connect()
