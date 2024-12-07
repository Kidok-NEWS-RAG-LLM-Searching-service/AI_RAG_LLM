from http.client import HTTPException

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.service.rag_pipeline import rag_pipeline

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    rag_result: str
    sources: list


@router.post("/query", response_model=QueryResponse)
async def get_query_result(request: QueryRequest):
    try:
        result = rag_pipeline.query(request.query)
        return result
    except Exception as e:
        # 에러 핸들링
        raise HTTPException(status_code=500, detail=str(e))