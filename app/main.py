import json
import logging
import os
import sys
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.middleware.base import BaseHTTPMiddleware

from app.api import router
from app.ml.predict import get_artifact
from app.queries import open_session

ACCESS_LOGGER = "app.access"


def database_check(connect: Callable[[], None]) -> str:
    try:
        connect()
    except Exception:  # noqa: BLE001 — any failure means the database check is down
        return "down"
    return "ok"


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            _access_logger().info(
                json.dumps(
                    {
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": status_code,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    }
                )
            )


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        get_artifact()
        yield

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(AccessLogMiddleware)
    app.include_router(router)

    @app.get("/health")
    def health() -> dict[str, object]:
        def ping() -> None:
            with open_session() as session:
                session.execute(text("SELECT 1"))

        database = database_check(ping)
        return {
            "status": "ok" if database == "ok" else "degraded",
            "version": os.environ.get("APP_VERSION", "0.1.0"),
            "checks": {"database": database},
        }

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail and "message" in detail:
            error = {"code": detail["code"], "message": detail["message"]}
        else:
            error = {"code": "http_error", "message": str(detail)}
        return JSONResponse(status_code=exc.status_code, content={"error": error})

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {"msg": "Invalid request", "loc": []}
        location = ".".join(str(part) for part in first.get("loc", []) if part not in {"query", "body"})
        message = str(first.get("msg", "Invalid request"))
        if location:
            message = f"{location}: {message}"
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": message}},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Internal server error"}},
        )

    return app


def _access_logger() -> logging.Logger:
    logger = logging.getLogger(ACCESS_LOGGER)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


app = create_app()
