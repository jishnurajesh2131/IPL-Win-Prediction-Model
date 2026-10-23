"""
Main FastAPI Application Entry Point for IPL Live Win Prediction Backend.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.backend.config import (
    API_PREFIX,
    HOST,
    PORT,
    PROJECT_TITLE,
    PROJECT_DESCRIPTION,
    PROJECT_VERSION,
)
from src.backend.api import api_router
from src.backend.services.match_service import match_service
from src.backend.services.simulation_service import simulation_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events: pre-warm datasets and verify ML artifacts on startup.
    """
    print("=" * 60)
    print(f"Starting {PROJECT_TITLE} (v{PROJECT_VERSION})")
    print("=" * 60)

    # Pre-load match metadata
    match_service._ensure_loaded()
    total_m = match_service.total_count()
    print(f"Match catalog loaded: {total_m:,} matches available.")

    # Pre-warm simulator data
    simulation_service.simulator._ensure_data_loaded()
    print("Frozen Two-Model live simulator engine ready.")

    # Pre-warm Prediction Intelligence data
    from src.backend.services.prediction_intelligence_service import prediction_intelligence_service
    prediction_intelligence_service.ensure_loaded()
    print("Prediction Intelligence Module (Auditor, Observatory, Era-Shift) ready.")

    yield

    print("Shutting down IPL Live Win Prediction Backend.")


app = FastAPI(
    title=PROJECT_TITLE,
    description=PROJECT_DESCRIPTION,
    version=PROJECT_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Enable CORS for local frontend integration (Vite on 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes under /api/v1
app.include_router(api_router, prefix=API_PREFIX)


@app.get(
    "/",
    tags=["Root"],
    summary="Root Service Info",
    description="Returns service metadata and navigation links.",
)
def root():
    return {
        "service": PROJECT_TITLE,
        "version": PROJECT_VERSION,
        "status": "online",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "api_v1_endpoints": {
            "health": f"{API_PREFIX}/health",
            "matches": f"{API_PREFIX}/matches",
            "simulations": f"{API_PREFIX}/simulations",
            "intelligence": f"{API_PREFIX}/intelligence",
        },
    }



# Global Exception Handlers
@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Not Found", "detail": str(exc).strip("'\"")},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Bad Request", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.backend.main:app", host=HOST, port=PORT, reload=True)
