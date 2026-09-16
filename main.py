from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from middleware.logging import StructuredLoggingMiddleware
from api.v1 import health, chat

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json"
    )

    # Middleware Configuration
    app.add_middleware(StructuredLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API Routers
    app.include_router(health.router, prefix=f"{settings.API_V1_STR}", tags=["health"])
    app.include_router(chat.router, prefix=f"{settings.API_V1_STR}", tags=["chat"])

    return app

app = create_app()

@app.get("/", summary="Root Endpoint")
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME} API. Visit /docs for the API documentation."}
