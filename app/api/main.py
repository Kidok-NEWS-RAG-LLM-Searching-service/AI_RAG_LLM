from fastapi import APIRouter

from app.api.routes import health_check_controller, ai_chat_controller

api_router = APIRouter()

api_router.include_router(health_check_controller.router, prefix="/health-check", tags=["Health Check"])
api_router.include_router(ai_chat_controller.router, prefix="/ai-chat", tags=["AI Chat"])