from http.client import HTTPException

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.service.rag_pipeline import rag_pipeline

router = APIRouter()

from fastapi.responses import StreamingResponse
from langchain_core.documents import Document
from typing import List


router = APIRouter()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    rag_result: str
    sources: list


class SourceResponse(BaseModel):
    docs: List[Document]
    sources: List[dict]


class DocsRequest(BaseModel):
    query: str
    docs: List[Document]


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
        sources = rag_pipeline.get_source(docs)
        result = {
            'docs': docs,
            'sources': sources
        }
        return result
    except Exception as e:
        print(e)
        # 에러 핸들링
        # raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream_result", response_class=StreamingResponse)
async def get_stream_result(request: DocsRequest):
    try:
        return StreamingResponse(
            rag_pipeline.stream_query(request.query, request.docs),
            media_type="text/event-stream",
        )
    except Exception as e:
        print(e)
        return StreamingResponse(
            (f"Error: {str(e)}\n" for _ in range(1)),
            media_type="text/event-stream",
        )


