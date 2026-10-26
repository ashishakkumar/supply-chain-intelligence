"""Process entry for the API.

Wires the routes, the JSON access log, the health check, and the error handlers.
Startup loads the delay-model artefact once. It does not seed or empty Postgres.
"""

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

ACCESS_LOGGER = "app.access" # purpose for this logger is to log the access logs (this means the requests to the server being made by the client)




def database_check(connect: Callable[[], None]) -> str:
    """Readiness probe for GET /health.

    Returns "ok" when the callable can reach Postgres, otherwise "down".
    Any failure is "down". The HTTP status of /health stays 200 either way,
    so a database blip is not treated as a dead process.
    """
    try:
        connect()
    except Exception:  # noqa: BLE001 — any failure means the database check is down
        return "down"
    return "ok"


class AccessLogMiddleware(BaseHTTPMiddleware):
    """One JSON access line per request.

    A middleware runs around the route so every response, including errors,
    records method, path, status code, and latency. Those four fields are the
    minimum the brief asks for.
    """

    async def dispatch(self, request: Request, call_next):
        """Time the route and log method, path, status code, and latency_ms."""
        started = time.perf_counter() # this is the time when the request started
        status_code = 500 # this is the status code of the response
        try:
            response = await call_next(request) # this is the response from the next middleware
            status_code = response.status_code # this is the status code of the response
            return response
        finally:
            _access_logger().info( # this is the logger for the access logs
                json.dumps(
                    {
                        "method": request.method, # this is the method of the request
                        "path": request.url.path, # this is the path of the request
                        "status_code": status_code, # this is the status code of the response
                        "latency_ms": round((time.perf_counter() - started) * 1000, 3), # this is the latency of the request
                    }
                )
            )


def create_app() -> FastAPI:
    """Build the FastAPI app: model load, access log, routes, and error JSON.

    The predict endpoint loads the artefact at startup, not on
    each request and not by retraining. The lifespan hook does that load once.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Load the delay model once when the process starts."""
        get_artifact() # this is the function to get the artifact
        yield

    app = FastAPI(lifespan=lifespan) # this is the application (lifepan is required because we are using the async context manager)
    app.add_middleware(AccessLogMiddleware) # this is the middleware to add the access log middleware
    app.include_router(router) # this is the router to include the router

    @app.get("/health")
    def health() -> dict[str, object]:
        """GET /health — liveness plus a database readiness check.

        Returns service status, version, and checks.database. status is "ok"
        only when the database check is "ok"; otherwise "degraded". The HTTP
        status stays 200. After a deploy, this is how you confirm the process
        should receive traffic (production support runbook).
        """

        def ping() -> None:
            """Run SELECT 1. A failure means the database check is down."""
            with open_session() as session: # this is the session to interact with the database
                session.execute(text("SELECT 1")) # this is the query to execute

        database = database_check(ping) # this is the database check
        return {
            "status": "ok" if database == "ok" else "degraded", # this is the status of the database
            "version": os.environ.get("APP_VERSION", "0.1.0"), # this is the version of the application
            "checks": {"database": database}, # this is the checks of the database
        }

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_request: Request, exc: HTTPException) -> JSONResponse:
        """Return {"error": {"code", "message"}} for raised HTTP errors.

        Routes that already built a code and a message (404 shipment_not_found,
        503 model_unavailable) keep them. Any other HTTP error becomes http_error
        so the client still gets the same JSON shape.
        """
        detail = exc.detail# this is the detail of the exception
        if isinstance(detail, dict) and "code" in detail and "message" in detail:
            error = {"code": detail["code"], "message": detail["message"]}
        else:
            error = {"code": "http_error", "message": str(detail)}# this is the error that is returned when the request is not successful
        return JSONResponse(status_code=exc.status_code, content={"error": error})

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        """422 validation_error when the query or body fails the schema.

        FastAPI's default body is a list of field errors. This handler collapses
        the first error into the same code-and-message object as the other routes.
        """
        first = exc.errors()[0] if exc.errors() else {"msg": "Invalid request", "loc": []}# this is the first error of the exception
        location = ".".join(str(part) for part in first.get("loc", []) if part not in {"query", "body"})# this is the location of the error
        message = str(first.get("msg", "Invalid request"))# this is the message of the error
        if location:
            message = f"{location}: {message}"# this is the message of the error
        return JSONResponse(
            status_code=422, # this is the status code of the response
            content={"error": {"code": "validation_error", "message": message}},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_request: Request, _exc: Exception) -> JSONResponse:
        """500 internal_error for anything the route did not handle.

        The body stays the structured error object. The original exception is
        not copied into the response.
        """
        return JSONResponse(
            status_code=500, # this is the status code of the response
            content={"error": {"code": "internal_error", "message": "Internal server error"}},
        )

    return app


def _access_logger() -> logging.Logger:
    """Stdout logger that prints the access JSON as the whole line.

    The formatter is the message only, so the line is one JSON object:
    method, path, status_code, latency_ms. propagate is off so uvicorn does
    not print the same line a second time.
    """
    logger = logging.getLogger(ACCESS_LOGGER)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout) # this is the handler for the logging
        handler.setFormatter(logging.Formatter("%(message)s")) # this is the formatter for the logging
        logger.addHandler(handler) # this is the handler for the logging
        logger.setLevel(logging.INFO) # this is the level of the logging
        logger.propagate = False # this is the propagate of the logging
    return logger


app = create_app() # this is the application


# this is not the cli, this is the main file for the application (the backend that runs the application)