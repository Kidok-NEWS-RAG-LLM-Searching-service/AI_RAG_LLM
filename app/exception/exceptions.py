from fastapi import HTTPException

from app.exception.error_code import ErrorCode


class BaseCustomException(HTTPException):
    """Base class for custom exceptions"""

    def __init__(self, status_code: int, code: str, detail: str):
        super().__init__(status_code=status_code, detail=detail)
        self.code = code

    def __str__(self):
        return "[" + self.code + "]" + " " + self.detail


class UnAuthorizeException(BaseCustomException):
    def __init__(
            self,
            status_code: int = ErrorCode.UNAUTHORIZED_ERROR.status_code,
            code: str = ErrorCode.UNAUTHORIZED_ERROR.code,
            detail: str = ErrorCode.UNAUTHORIZED_ERROR.message
    ):
        super().__init__(
            status_code=status_code,
            code=code,
            detail=detail
        )


class UnSupportedContentTypeException(BaseCustomException):
    def __init__(self):
        super().__init__(
            status_code=ErrorCode.UNSUPPORTED_CONTENT_TYPE_ERROR.status_code,
            code=ErrorCode.UNSUPPORTED_CONTENT_TYPE_ERROR.code,
            detail=ErrorCode.UNSUPPORTED_CONTENT_TYPE_ERROR.message
        )


class NotFoundAuthorizationHeaderException(BaseCustomException):
    def __init__(self):
        super().__init__(
            status_code=ErrorCode.NOT_FOUND_AUTHORIZATION_HEADER.status_code,
            code=ErrorCode.NOT_FOUND_AUTHORIZATION_HEADER.code,
            detail=ErrorCode.NOT_FOUND_AUTHORIZATION_HEADER.message
        )
