"""
API routes package registering all endpoint routers.
"""

from fastapi import APIRouter

from .health import router as health_router
from .matches import router as matches_router
from .simulation import router as simulation_router
from .intelligence import router as intelligence_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(matches_router)
api_router.include_router(simulation_router)
api_router.include_router(intelligence_router)

__all__ = ["api_router"]

