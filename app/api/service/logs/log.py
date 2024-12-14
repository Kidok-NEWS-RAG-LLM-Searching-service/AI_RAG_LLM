from app.api.repository.ai_search_log_repository import ai_search_log_repository
from app.api.repository.ai_search_log_type import AISearchLogType


@DeprecationWarning
def read():
    items = ai_search_log_repository.get_all()
    print("items", items)


@DeprecationWarning
def put():
    ai_search_log_repository.put_item(
        log_type=AISearchLogType.SEARCH,
        query="test",
    )


def get_item(column_name, value):
    return ai_search_log_repository.get_by_column(column_name=column_name, value=value)
