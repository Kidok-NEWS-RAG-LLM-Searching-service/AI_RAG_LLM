from fastapi import FastAPI, Depends
from fastapi.security import APIKeyHeader
from starlette.middleware.cors import CORSMiddleware

from app.api.main import api_router
from app.core.config import settings

auth_header = APIKeyHeader(name="Authorization", auto_error=False)

app = FastAPI(dependencies=[Depends(auth_header)])

app.include_router(api_router, prefix=settings.API_V1_STR)

if settings.all_cors_origin:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origin,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
