import time
from http.client import HTTPException

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.repository.ai_model_performance_repository import ai_model_performance_log_repository
from app.api.repository.cache.cache_repository import cache_repository
from app.api.service.cache.cache_service import cache_service
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
        cache_data = await cache_repository.get_today_cache_data()
        cache_information = await rag_pipeline.is_cache_hit(request.query, cache_data)
        if cache_information.get("hit_check"):
            answer = cache_information.get("answer")
            sources_list = cache_information.get("sources_list")
            response = {
                "rag_result": answer,
                "sources": sources_list
            }
            end_cache_time = time.time()
            await ai_model_performance_log_repository.put_item(
                answer_model_type=cache_information.get("answer_model_type"),
                intent_model_type=cache_information.get("intent_model_type"),
                config=cache_information.get("config"),
                query=request.query,
                answer=cache_information.get("answer"),
                get_document_start_timestamp=-1,
                get_document_end_timestamp=-1,
                len_document=-1,
                model_duration=cache_information.get("model_duration"),
                query_routing_start_timestamp=start_log_time,
                query_routing_end_timestamp=end_cache_time,
                id_list=cache_information.get("origin_id_list"),
                remove_duplicates_id_list=cache_information.get("remove_duplicates_id_list"),
                check_id_list=cache_information.get("hallucination_check"),
                wrong_sources=cache_information.get("hallucination_in_llm_response"),
                deleted_ids_list=cache_information.get("deleted_sources_in_llm"),
                cache_information={
                    "has_cache_hit": 1,
                    "quoted_query": cache_information.get("query")
                }
            )
            await put_search_response_tracking(query=request.query, answer=answer)

            return response

        result, intent = await rag_pipeline.query_model_pipeline(request.query)

        answer, sources_list, making_sources, remove_hallucinated_sources = await rag_pipeline.get_answer(result)
        end_log_time = time.time()

        # cache miss
        await ai_model_performance_log_repository.put_item(
            answer_model_type=result.get("model_type"),
            intent_model_type=intent,
            config=result.get("config"),
            query=request.query,
            answer=answer,
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
            deleted_ids_list=remove_hallucinated_sources.get("deleted_ids_list"),
            cache_information={
                "has_cache_hit": 0,
                "quoted_query": "None"
            }
        )

        # intent_model_type == answer_model_type &&
        # (len_hallucination_check_pass/len_remove_duplicates_id_list) >= 0.8
        # print("confirm : ", result.get("model_type")[:8] == intent[:8], making_sources.get("check_id_list").count("PASS") / len(making_sources.get("check_id_list")) >= 0.8)
        if result.get("model_type")[:8] == intent[:8] and \
            len(making_sources.get("check_id_list")) != 0 and \
            making_sources.get("check_id_list").count("PASS") != 0 and \
            making_sources.get("check_id_list").count("PASS") / len(making_sources.get("check_id_list")) >= 0.8:
            await cache_repository.put_item(
                answer_model_type=result.get("model_type"), # 삭제해도 될듯
                intent_model_type=intent,
                config=result.get("config"),
                query=request.query,
                query_embed=cache_information.get("query_embed"),
                answer=answer,
                get_document_start_timestamp=result.get("get_document_start_timestamp"),
                get_document_end_timestamp=result.get("get_document_end_timestamp"),
                len_document=result.get("document_length"),
                model_duration=result.get("model_duration"),
                id_list=making_sources.get("id_list"),
                remove_duplicates_id_list=making_sources.get("remove_duplicates_id_list"),
                check_id_list=making_sources.get("hallucination_check"),
                wrong_sources=remove_hallucinated_sources.get("wrong_sources"),
                deleted_ids_list=remove_hallucinated_sources.get("deleted_ids_list"),
                sources_list=sources_list
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


@router.get("/mongo")
async def get_query_result():
    await cache_repository.get_today_cache_data()
    return "success"









