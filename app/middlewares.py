import time
import logging
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

from app.logging_conf import setup_logging


# --------------------------------------------------
# Logging setup
# --------------------------------------------------
setup_logging()
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Middleware: Request ID + timing
# --------------------------------------------------
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception as e:
            logger.exception(
                "Unhandled error",
                extra={"extra_fields": {"request_id": request_id, "error": str(e)}}
            )
            raise e

        process_time = time.perf_counter() - start_time

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}"

        logger.info(
            "Request completed",
            extra={"extra_fields": {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time": round(process_time, 4),
            },
        })

        return response
