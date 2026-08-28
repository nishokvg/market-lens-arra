from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlparse

from langchain_core.tools import tool


DEMO_COMPETITORS = ["Demo Apparel Supply", "Demo Uniform Works", "Demo Activewear Co"]

DEMO_SOURCES = {
    "Demo Apparel Supply": [
        {
            "title": "Demo Apparel Supply - Custom Manufacturing",
            "url": "https://example.com/demo-apparel-supply",
            "content": "Sample evidence for classroom use only. Demo Apparel Supply offers custom cotton T-shirts and polos. Typical sample MOQ is 500 units. Production is listed as Vietnam with estimated lead time of 45 to 60 days.",
        }
    ],
    "Demo Uniform Works": [
        {
            "title": "Demo Uniform Works - Services",
            "url": "https://example.com/demo-uniform-works",
            "content": "Sample evidence for classroom use only. Demo Uniform Works focuses on corporate uniforms and embroidery customization. Its sample MOQ is 250 units. It lists production in India and shipping to the United States.",
        }
    ],
    "Demo Activewear Co": [
        {
            "title": "Demo Activewear Co - Capabilities",
            "url": "https://example.com/demo-activewear",
            "content": "Sample evidence for classroom use only. Demo Activewear Co produces sportswear and hoodies. It advertises recycled polyester material options and private-label customization. The sample lead time is 30 to 45 days.",
        }
    ],
}

EXCLUDED_DISCOVERY_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "reddit.com",
    "youtube.com",
    "wikipedia.org",
    "yelp.com",
    "foursource.com",
    "manufacturer.clothing",
}

GENERIC_COMPETITOR_LABELS = {
    "apparel",
    "apparel manufacturer",
    "apparel manufacturers",
    "clothing manufacturer",
    "clothing manufacturers",
    "custom apparel",
    "custom clothing",
    "manufacturers",
    "suppliers",
}

GENERIC_TITLE_WORDS = {
    "apparel",
    "clothing",
    "custom",
    "manufacturer",
    "manufacturers",
    "sportswear",
    "supplier",
    "suppliers",
    "uniform",
    "uniforms",
    "company",
    "companies",
    "corporate",
    "business",
    "print",
    "prints",
    "made",
    "usa",
    "for",
    "growing",
    "brands",
    "t",
    "shirt",
    "shirts",
    "polos",
    "tops",
    "united",
    "states",
}

LISTICLE_PREFIXES = ("top ", "best ", "leading ", "list of ", "guide to ")


def _tavily_search(query: str, max_results: int) -> dict[str, Any]:
    """Run a Tavily search behind one seam so provider behavior is easy to test."""
    from tavily import TavilyClient

    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=True,
    )


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _domain_name(url: str) -> str:
    hostname = urlparse(url).hostname or ""
    labels = [label for label in hostname.lower().split(".") if label not in {"www", "com", "net", "org", "io", "co", "us"}]
    if not labels:
        return ""
    name = labels[0].replace("-", " ")
    for suffix in ("apparel", "clothing", "uniforms", "sportswear", "industries", "manufacturing"):
        if name.endswith(suffix) and name != suffix and f" {suffix}" not in name:
            name = f"{name[:-len(suffix)]} {suffix}"
            break
    return " ".join(name.split()).title()


def _candidate_from_result(result: dict[str, Any]) -> str:
    """Prefer a company-like title segment and fall back to the result domain."""
    title = result.get("title", "")
    parts = [part.strip() for part in re.split(r"[|:–—-]", title) if part.strip()]
    for part in parts:
        normalized = _normalize(part)
        words = set(normalized.split())
        if (
            normalized
            and normalized not in GENERIC_COMPETITOR_LABELS
            and not normalized.startswith(LISTICLE_PREFIXES)
            and not words.issubset(GENERIC_TITLE_WORDS)
            and len(normalized.split()) <= 6
        ):
            return part
    return _domain_name(result.get("url", ""))


def _is_excluded_result(result: dict[str, Any], target_company: str) -> bool:
    hostname = (urlparse(result.get("url", "")).hostname or "").lower()
    if any(hostname == domain or hostname.endswith(f".{domain}") for domain in EXCLUDED_DISCOVERY_DOMAINS):
        return True
    target_words = set(_normalize(target_company).split())
    candidate_words = set(_normalize(_candidate_from_result(result)).split())
    return bool(target_words and candidate_words and target_words.issubset(candidate_words))


@tool
def discover_competitors(
    company_name: str,
    target_market: str,
    product_categories: list[str],
    competitor_limit: int = 10,
) -> list[str]:
    """Find up to ten apparel competitors with Tavily; demo mode returns labelled sample companies."""
    competitor_limit = max(3, min(competitor_limit, 10))
    if os.getenv("DEMO_MODE", "true").lower() == "true":
        return DEMO_COMPETITORS[:competitor_limit]

    if not os.getenv("TAVILY_API_KEY"):
        return []

    categories = ", ".join(product_categories)
    query = (
        f"{categories} custom apparel manufacturer and supplier in {target_market} official website; "
        "corporate uniforms, private-label apparel, and sportswear"
    )
    try:
        # Ask for extra candidates because directory and target-company results are filtered out.
        response = _tavily_search(query, max_results=min(20, competitor_limit * 2))
    except Exception:
        return []

    competitors: list[str] = []
    seen: set[str] = set()
    for result in response.get("results", []):
        if _is_excluded_result(result, company_name):
            continue
        candidate = _candidate_from_result(result)
        key = _normalize(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        competitors.append(candidate)
        if len(competitors) == competitor_limit:
            break
    return competitors


@tool
def research_competitor(competitor: str, target_market: str, requested_fields: list[str]) -> list[dict[str, str]]:
    """Collect fresh public evidence for one competitor with Tavily, or deterministic demo evidence."""
    if os.getenv("DEMO_MODE", "true").lower() == "true":
        return DEMO_SOURCES.get(competitor, [])

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    query_fields = ", ".join(requested_fields)
    query = f"{competitor} apparel manufacturer {target_market} {query_fields}"
    try:
        response = _tavily_search(query, max_results=5)
    except Exception:
        return []
    return [
        {"title": result.get("title", "Untitled"), "url": result["url"], "content": result.get("content", "")}
        for result in response.get("results", [])
    ]
