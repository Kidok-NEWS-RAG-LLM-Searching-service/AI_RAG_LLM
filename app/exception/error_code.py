from enum import Enum


class StatusCode:
    OK_SUCCESS: int = 200

    BAD_REQUEST: int = 400
    UNAUTHORIZED: int = 401
    FORBIDDEN: int = 403
    NOT_FOUND: int = 404

    INTERNAL_SERVER_ERROR: int = 500


class ErrorCode(Enum):

    """""""""""""""""""""""
    ""       COMMON      ""
    """""""""""""""""""""""
    HEALTH_CHECK_SERVER_ERROR = (StatusCode.INTERNAL_SERVER_ERROR, "HEALTH_CHECK_SERVER_ERROR", "Health Check Error")
    UNAUTHORIZED_ERROR = (StatusCode.UNAUTHORIZED, "UNAUTHORIZED_ERROR", "Unauthorized access")
    UNSUPPORTED_CONTENT_TYPE_ERROR = (StatusCode.BAD_REQUEST, "UNSUPPORTED_CONTENT_TYPE_ERROR", "지원하지 않는 Content Type 입니다.")

    """""""""""""""""""""""
    ""        AUTH       ""
    """""""""""""""""""""""
    EMPTY_JWT_TOKEN_ERROR = (StatusCode.UNAUTHORIZED, "EMPTY_JWT_TOKEN_ERROR", "JWT token is missing")
    NOT_FOUND_AUTHORIZATION_HEADER = (StatusCode.UNAUTHORIZED, "NOT_FOUND_AUTHORIZATION_HEADER", "Authorization 값이 Header에 존재하지 않습니다.")

    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message
