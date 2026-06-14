"""FastAPI exception handlers for Postgres read failures."""

from fastapi import Request
from fastapi.responses import JSONResponse

from cacheon_db.exceptions import DatabaseNotConfigured, DatabaseUnavailable


async def database_not_configured_handler(
    _request: Request, _exc: DatabaseNotConfigured
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "CACHEON_DATABASE_URL is not configured"},
    )


async def database_unavailable_handler(
    _request: Request, _exc: DatabaseUnavailable
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "database connection failed"},
    )
