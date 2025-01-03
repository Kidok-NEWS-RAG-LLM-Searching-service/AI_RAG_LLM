import time
from http.client import HTTPException

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.repository.ai_model_performance_repository import ai_model_performance_log_repository
from app.api.service.logs.log import put_search_response_tracking
from app.api.service.rag_pipeline import rag_pipeline

router = APIRouter()

from fastapi.responses import StreamingResponse
from langchain_core.documents import Document
from typing import List


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


# @router.post("/query_routing", response_model=QueryResponse)
# async def get_query_result(request: QueryRequest):
#     try:
#         result = rag_pipeline.query_model_pipeline(request.query)
#         answer, sources_list = rag_pipeline.get_answer(result)
#         sources = rag_pipeline.makeing_source(result, sources_list)

#         response = {
#             "rag_result": answer,
#             "sources": sources
#         }
#         await put_search_response_tracking(query=request.query, answer=answer)

#         return response
#     except Exception as e:
#         # 에러 핸들링
#         raise HTTPException(500, str(e))

# 비동기적으로 처리
@router.post("/query_routing", response_model=QueryResponse)
async def get_query_result(request: QueryRequest):
    try:
        start_log_time = time.time()
        result = await rag_pipeline.query_model_pipeline(request.query)

        answer, sources_list, making_sources, remove_hallucinated_sources = await rag_pipeline.get_answer(result)
        end_log_time = time.time()

        await ai_model_performance_log_repository.put_item(
            model_type=result.get("model_type"),
            config=result.get("config"),
            get_document_start_timestamp=result.get("get_document_start_timestamp"),
            get_document_end_timestamp=result.get("get_document_end_timestamp"),
            len_document=result.get("document_length"),
            model_duration=result.get("model_duration"),
            query_routing_start_timestamp=start_log_time,
            query_routing_end_timestamp=end_log_time,
            id_list=making_sources.get("id_list"),
            remove_duplicates_id_list=making_sources.get("remove_duplicates_id_list"),
            check_id_list=making_sources.get("check_id_list"),
            wrong_sources=remove_hallucinated_sources.get("wrong_sources"),
            deleted_ids_list=remove_hallucinated_sources.get("deleted_ids_list")
        )

        response = {
            "rag_result": answer,
            "sources": sources_list
        }
        await put_search_response_tracking(query=request.query, answer=answer)

        return response
    except Exception as e:
        # 에러 핸들링
        raise HTTPException(500, str(e))










