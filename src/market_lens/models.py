from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchBrief(BaseModel):
    company_name: str
    target_market: str
    product_categories: list[str]
    requested_fields: list[str]
    known_competitors: list[str] = Field(default_factory=list)
    competitor_limit: int = Field(default=10, ge=3, le=10)
    user_id: str = "arra-global-demo"


class Evidence(BaseModel):
    url: str
    quote: str
    source_title: str


class CompetitorFact(BaseModel):
    competitor: str
    field: str
    value: str = "Unknown"
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = 0.0
