from fastapi import APIRouter

from app.api.routes import health_check_controller

api_router = APIRouter()

api_router.include_router(health_check_controller.router, prefix="/health-check", tags=["Health Check"])