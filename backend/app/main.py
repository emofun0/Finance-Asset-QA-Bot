from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.api.routes import api_router
from app.core.config import settings
from app.core.errors import AppError
from app.schemas.response import ErrorDetail, StandardResponse

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Financial Asset QA System backend service.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://127.0.0.1:{settings.frontend_port}",
        f"http://localhost:{settings.frontend_port}",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.exception_handler(AppError)
def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
    body = StandardResponse(
        request_id=str(uuid4()),
        success=False,
        error=ErrorDetail(
            code=exc.code,
            message=exc.message,
            details=exc.details,
        ),
    )
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


@app.exception_handler(RequestValidationError)
def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    body = StandardResponse(
        request_id=str(uuid4()),
        success=False,
        error=ErrorDetail(
            code="VALIDATION_ERROR",
            message="请求参数不合法。",
            details={"errors": exc.errors()},
        ),
    )
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=body.model_dump())


@app.exception_handler(Exception)
def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    body = StandardResponse(
        request_id=str(uuid4()),
        success=False,
        error=ErrorDetail(
            code="INTERNAL_SERVER_ERROR",
            message="服务器内部错误。",
            details={"type": exc.__class__.__name__},
        ),
    )
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=body.model_dump())


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Finance Asset QA System backend is running.",
    }
