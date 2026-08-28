from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from market_lens.integrations import FactExtractionService, MemoryService, SourceKnowledgeBase
from market_lens.models import CompetitorFact, Evidence, ResearchBrief
from market_lens.research_tools import discover_competitors, research_competitor


class ResearchState(TypedDict, total=False):
    brief: dict[str, Any]
    memory_context: list[dict[str, Any]]
    retrieved_sources: list[dict[str, Any]]
    integration_status: dict[str, str]
    competitors: list[str]
    sources: dict[str, list[dict[str, str]]]
    facts: list[dict[str, Any]]
    comparison_rows: list[dict[str, str]]
    coverage_score: float
    review_notes: list[str]
    draft_report: str
    errors: list[str]


KEYWORDS = {
    "moq": ["moq", "minimum order"],
    "pricing": ["price", "pricing", "cost"],
    "materials": ["cotton", "polyester", "material", "fabric"],
    "customization": ["custom", "private-label", "embroidery", "printing"],
    "certifications": ["certificate", "certified", "gots", "oeko"],
    "production location": ["production", "manufacturing", "made in", "vietnam", "india", "china"],
    "lead time": ["lead time", "days", "weeks"],
    "US delivery": ["united states", "us delivery", "shipping"],
}

WORKFLOW_STAGE_LABELS = {
    "load_context": "Loading saved research context",
    "discover": "Discovering competitors",
    "research": "Collecting public competitor evidence",
    "index_sources": "Indexing sources for future retrieval",
    "extract": "Extracting comparable facts",
    "analyze": "Building the competitor comparison",
    "review": "Reviewing evidence coverage and risks",
}


def _research_worker_count(competitor_count: int) -> int:
    """Keep concurrent provider calls bounded even when configuration is malformed."""
    try:
        configured_workers = int(os.getenv("MAX_RESEARCH_WORKERS", "3"))
    except ValueError:
        configured_workers = 3
    return max(1, min(configured_workers, competitor_count))


def context_node(state: ResearchState) -> ResearchState:
    brief = ResearchBrief.model_validate(state["brief"])
    query = f"{brief.company_name} {brief.target_market} {', '.join(brief.product_categories)}"
    memory = MemoryService().recall(brief.user_id, query)
    knowledge = SourceKnowledgeBase().retrieve(query)
    return {
        "memory_context": memory.items,
        "retrieved_sources": knowledge.items,
        "integration_status": {"mem0": memory.status, "pinecone": knowledge.status},
    }


def discover_node(state: ResearchState) -> ResearchState:
    brief = ResearchBrief.model_validate(state["brief"])
    competitors: list[str] = []
    seen: set[str] = set()
    for competitor in brief.known_competitors:
        key = competitor.strip().lower()
        if key and key not in seen:
            competitors.append(competitor.strip())
            seen.add(key)
    if len(competitors) < brief.competitor_limit:
        discovered = discover_competitors.invoke(
            {
                "company_name": brief.company_name,
                "target_market": brief.target_market,
                "product_categories": brief.product_categories,
                "competitor_limit": brief.competitor_limit,
            }
        )
        for competitor in discovered:
            key = competitor.strip().lower()
            if key and key not in seen:
                competitors.append(competitor.strip())
                seen.add(key)
            if len(competitors) == brief.competitor_limit:
                break
    return {
        "competitors": competitors[: brief.competitor_limit],
        "errors": [] if competitors else ["No competitors were discovered. Add known competitors or enable demo mode."],
    }


def research_node(state: ResearchState) -> ResearchState:
    brief = ResearchBrief.model_validate(state["brief"])
    sources: dict[str, list[dict[str, str]]] = {}
    errors = list(state.get("errors", []))
    competitors = state.get("competitors", [])

    def collect(competitor: str) -> tuple[str, list[dict[str, str]]]:
        results = research_competitor.invoke(
            {"competitor": competitor, "target_market": brief.target_market, "requested_fields": brief.requested_fields}
        )
        return competitor, results

    max_workers = _research_worker_count(len(competitors))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        collected = list(executor.map(collect, competitors))
    for competitor, results in collected:
        sources[competitor] = results
        if not results:
            errors.append(f"No public evidence found for {competitor}.")
    return {"sources": sources, "errors": errors}


def index_sources_node(state: ResearchState) -> ResearchState:
    result = SourceKnowledgeBase().index_sources(state.get("sources", {}))
    statuses = dict(state.get("integration_status", {}))
    statuses["pinecone_indexing"] = result.status
    errors = list(state.get("errors", []))
    if result.status == "unavailable":
        errors.append(f"Source indexing unavailable: {result.message}")
    return {"integration_status": statuses, "errors": errors}


def _extract_fact(competitor: str, field: str, source: dict[str, str]) -> CompetitorFact:
    content = source.get("content", "")
    keywords = KEYWORDS.get(field.lower(), [field.lower()])
    matching_sentence = next(
        (sentence.strip() for sentence in content.split(".") if any(keyword in sentence.lower() for keyword in keywords)),
        "",
    )
    if not matching_sentence:
        return CompetitorFact(competitor=competitor, field=field)
    return CompetitorFact(
        competitor=competitor,
        field=field,
        value=matching_sentence,
        confidence=0.65,
        evidence=[Evidence(url=source["url"], quote=matching_sentence, source_title=source["title"])],
    )


def extract_node(state: ResearchState) -> ResearchState:
    brief = ResearchBrief.model_validate(state["brief"])
    facts: list[dict[str, Any]] = []
    statuses = dict(state.get("integration_status", {}))
    extraction_service = FactExtractionService()
    for competitor, competitor_sources in state.get("sources", {}).items():
        extracted = extraction_service.extract(competitor, brief.requested_fields, competitor_sources)
        statuses["llm_extraction"] = extracted.status
        extracted_by_field = {item["field"].strip().lower(): item for item in extracted.items}
        for field in brief.requested_fields:
            llm_fact = extracted_by_field.get(field.strip().lower())
            if llm_fact:
                evidence = [Evidence.model_validate(item) for item in llm_fact.get("evidence", [])]
                fact = CompetitorFact(
                    competitor=competitor,
                    field=field,
                    value=llm_fact.get("value", "Unknown"),
                    confidence=0.85 if evidence else 0.0,
                    evidence=evidence,
                )
            else:
                fact = next(
                    (
                        candidate
                        for source in competitor_sources
                        if (candidate := _extract_fact(competitor, field, source)).evidence
                    ),
                    CompetitorFact(competitor=competitor, field=field),
                )
            facts.append(fact.model_dump())
    return {"facts": facts, "integration_status": statuses}


def analyze_node(state: ResearchState) -> ResearchState:
    brief = ResearchBrief.model_validate(state["brief"])
    facts = state.get("facts", [])
    rows: list[dict[str, str]] = []
    for competitor in state.get("competitors", []):
        row = {"Competitor": competitor}
        for field in brief.requested_fields:
            fact = next((item for item in facts if item["competitor"] == competitor and item["field"] == field), None)
            row[field] = fact["value"] if fact else "Unknown"
        rows.append(row)

    supported = sum(1 for fact in facts if fact["evidence"])
    coverage = supported / len(facts) if facts else 0.0
    draft = (
        f"MarketLens compared {len(rows)} competitors for {brief.company_name} in {brief.target_market}. "
        f"Evidence supports {supported} of {len(facts)} requested comparison points ({coverage:.0%}). "
        "Use the comparison table to validate commercial terms directly with each supplier before outreach."
    )
    return {"comparison_rows": rows, "coverage_score": coverage, "draft_report": draft}


def review_node(state: ResearchState) -> ResearchState:
    notes = list(state.get("errors", []))
    if state.get("coverage_score", 0.0) < 0.8:
        notes.append("Coverage is below the 80% target. Treat unsupported fields as follow-up questions, not conclusions.")
    brief = ResearchBrief.model_validate(state["brief"])
    if len(state.get("competitors", [])) < brief.competitor_limit:
        notes.append(
            f"Only {len(state.get('competitors', []))} of {brief.competitor_limit} requested competitors were available; the report is incomplete."
        )
    return {"review_notes": notes}


def build_research_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("load_context", context_node)
    graph.add_node("discover", discover_node)
    graph.add_node("research", research_node)
    graph.add_node("index_sources", index_sources_node)
    graph.add_node("extract", extract_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("review", review_node)
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "discover")
    graph.add_edge("discover", "research")
    graph.add_edge("research", "index_sources")
    graph.add_edge("index_sources", "extract")
    graph.add_edge("extract", "analyze")
    graph.add_edge("analyze", "review")
    graph.add_edge("review", END)
    return graph.compile()


def run_research_with_progress(brief: dict[str, Any], on_stage: Callable[[str], None]) -> ResearchState:
    """Run the graph while exposing completed stages to the Streamlit UI."""
    state: ResearchState = {"brief": brief}
    for update in build_research_graph().stream(state, stream_mode="updates"):
        for stage, changes in update.items():
            state.update(changes)
            on_stage(stage)
    return state
