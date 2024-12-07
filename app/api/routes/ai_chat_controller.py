from http.client import HTTPException
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.service.rag_pipeline import rag_pipeline

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    rag_result: str
    sources: list


class SourceResponse(BaseModel):
    docs: list
    source: list


class DocsRequest(BaseModel):
    query: str
    docs: Any


class StreamResponse(BaseModel):
    event: str


@router.post("/query", response_model=QueryResponse)
def get_query_result(request: QueryRequest):
    try:
        result = rag_pipeline.query(request.query)
        return result
    except Exception as e:
        # 에러 핸들링
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/docs", response_model=SourceResponse)
def get_source(request: QueryRequest):
    try:
        docs = rag_pipeline.get_documents(request.query)
        print('api_docs :', docs)
        source = rag_pipeline.get_source(docs)
        result = {
            'docs': docs,
            'source': source
        }
        return result
    except Exception as e:
        print(e)
        # 에러 핸들링
        # raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream_result", response_model=StreamResponse)
async def get_stream_result(request: DocsRequest):
    try:
        await rag_pipeline.stream_query(request.query, request.docs)
    except Exception as e:
        # 에러 핸들링
        print(e)


