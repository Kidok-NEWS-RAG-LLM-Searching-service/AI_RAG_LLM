import boto3
from boto3.resources.base import ServiceResource

from app.core.config import settings


log_db: ServiceResource = boto3.resource(
        'dynamodb',
        region_name=settings.log_db_region_name,
        aws_access_key_id=settings.log_db_access_key_id,
        aws_secret_access_key=settings.log_db_secret_access_key
    )
