from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class HealthCheck(BaseModel):
    status: str = "OK"
    version: str = "1.0.0"

@router.get("/health", response_model=HealthCheck, summary="Perform a Health Check")
async def health_check() -> HealthCheck:
    """
    Endpoint to perform a basic health check on the API.
    Returns 200 OK if the API is running.
    """
    return HealthCheck(status="OK")

@router.get("/ping", summary="Ping the API")
async def ping():
    return {"ping": "pong"}
