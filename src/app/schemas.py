"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel


class ScanResponse(BaseModel):
    """Response schema for skin scan analysis."""

    scores: dict[str, float]
    overlays: dict[str, str]  # base64 encoded PNG images
    regions: list[str]


class HealthResponse(BaseModel):
    """Health check response."""

    ok: bool
