from fastapi import FastAPI

from app.exception.exception_handler import ExceptionRegistry, UnSupportedContentTypeExceptionHandler, \
    NotFoundAuthorizationHeaderExceptionHandler, UnAuthorizeExceptionHandler


class ExceptionHandlerRegistrar:
    """
    Class to manage and register exception handlers with the registry.
    """

    def __init__(self, registry: ExceptionRegistry):
        self.registry = registry
        # Define the exception handlers to register
        self.exception_handlers = [
            UnSupportedContentTypeExceptionHandler,
            NotFoundAuthorizationHeaderExceptionHandler,
            UnAuthorizeExceptionHandler,
        ]

    def register_all_handlers(self):
        """
        Register all exception handlers defined in the `exception_handlers` list.
        """
        for handler in self.exception_handlers:
            self.registry.register(handler)

    def apply_to_app(self, app: FastAPI):
        """
        Apply all registered handlers to the FastAPI app.
        """
        self.registry.apply(app)


def register_exception_handlers(app: FastAPI):
    """
    Register and apply exception handlers to the FastAPI app.
    """
    exception_registry = ExceptionRegistry()
    registrar = ExceptionHandlerRegistrar(exception_registry)
    registrar.register_all_handlers()
    registrar.apply_to_app(app)
