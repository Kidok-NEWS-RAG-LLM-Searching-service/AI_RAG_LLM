from typing import Type

from starlette.requests import Request
from starlette.responses import JSONResponse

from app.decorators.exception_decorators import exception_log
from app.exception.exceptions import UnSupportedContentTypeException, NotFoundAuthorizationHeaderException, \
    UnAuthorizeException


class BaseExceptionHandler:
    """
    Base class for exception handlers.
    Provides common behavior for all handlers.
    """

    exception_cls: Type[Exception] = Exception  # This should be overridden in subclasses.

    @staticmethod
    @exception_log
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        return BaseExceptionHandler._get_exception_response(exc)

    @staticmethod
    def _get_exception_response(exc: Exception) -> JSONResponse:
        """
        Generates a standard JSON response for exceptions.
        """
        return JSONResponse(
            status_code=getattr(exc, "status_code", 500),
            content={
                "code": getattr(exc, "code", "INTERNAL_ERROR"),
                "message": getattr(exc, "detail", "An unexpected error occurred."),
                "type": exc.__class__.__name__,
            },
        )


class UnSupportedContentTypeExceptionHandler(BaseExceptionHandler):
    """
    Handler for UnSupportedContentTypeException.
    """
    exception_cls = UnSupportedContentTypeException


class NotFoundAuthorizationHeaderExceptionHandler(BaseExceptionHandler):
    """
    Handler for NotFoundAuthorizationHeaderException.
    """
    exception_cls = NotFoundAuthorizationHeaderException


class UnAuthorizeExceptionHandler(BaseExceptionHandler):
    """
    Handler for UnAuthorizeException.
    """
    exception_cls = UnAuthorizeException
